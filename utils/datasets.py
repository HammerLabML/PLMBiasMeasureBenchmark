import difflib
import string

import os
import datasets
import pandas as pd
from datasets import load_dataset

from embedding import BertHuggingfaceMLM
from transformers import PreTrainedTokenizer

from unmasking_bias import get_token_diffs

from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
import itertools
from collections.abc import Callable
import numpy as np
import math
import pickle
import time
from tqdm import tqdm
from sklearn.utils import resample as sklearn_resample

def resample(X: np.ndarray, y: np.ndarray, groups: list, add_noise=False):
    assert (len(X) == len(y) and len(y) == len(groups)), "inconsistent number of samples for X,y,groups: "+str(len(X))+","+str(len(y))+","+str(len(groups))
    assert (len(np.unique(y)) == np.max(y)+1 and len(set(groups)) == max(groups)+1), "some label or group is missing in the dataset/ in this split"
    n_classes = np.max(y)+1
    n_groups = max(groups)+1
    
    y_group = []
    ids = [i for i in range(len(y))]
    for i in range(len(y)):
        combined_y = y[i]*n_groups+groups[i]
        y_group.append(combined_y)
            
    ids, _ = sklearn_resample(ids,y_group)
    ids_done = []
    if add_noise:
        print("resample with noise...")
        var_X = np.std(X, axis=1)
        print(var_X.shape)
        
        X_new = []
        for i in ids:
            if i in ids_done:
                noise = np.asarray([np.random.normal(0,var_X[j]/20) for j in range(X.shape[1])])
                X_new.append(np.asarray(X[i])+noise)
            else:
                X_new.append(X[i])
        X = np.asarray(X_new)
    else:
        X = np.asarray([X[i] for i in ids])
    y = np.asarray([y[i] for i in ids])
    groups = [groups[i] for i in ids]
    print("data shape after resample:")
    print(X.shape)
    return X, y, groups
    

class BiasDataset():
    
    def __init__(self):
        self.data = []
        self.bias_types = []
        self.groups_by_bias_types = {}
        self.sel_bias_type = None
        self.sel_groups = []
        self.sel_data = []
        self.individual_biases = []
        self.bias_score = None # or class-wise?
        
    def sel_attributes(self, bias_type: str) -> bool:
        if not bias_type in self.bias_types:
            print("bias type", bias_type, "is not supported for this dataset")
            return False
        
        self.sel_bias_type = bias_type
        self.sel_groups = self.groups_by_bias_types[bias_type]
        
        self.sel_data = []
        for sample in self.data:
            if sample['bias_type'] == self.sel_bias_type:
                self.sel_data.append(sample)
        return True
    
    def get_neutral_samples_by_masking(self, tokenizer: PreTrainedTokenizer):
        pass
                    
    # the dataset/ task specific extrinsic bias measure
    def individual_bias(self, sample: dict):
        pass
        
    def group_bias(self):
        pass
        
    def compute_invidivual_extrinsic_biases(self):
        self.individual_biases = []
        for sample in self.sel_data:
            score = self.individual_bias(sample)
            self.individual_biases.append(score)
            
            
################################################
###########    CrowSPairs dataset    ###########
################################################

PUNCTUATION = string.punctuation.replace('-','')

def get_group_label(modified_terms: list, bias_type: str, groups_by_bias_types: dict, terms_by_groups: dict):
    if not bias_type in groups_by_bias_types.keys():
        return None, None
    assert len(modified_terms) > 0
    
    group_lbl = None
    terms_missing = {group: [] for group in groups_by_bias_types[bias_type]}
    for group in groups_by_bias_types[bias_type]:
        group_terms = terms_by_groups[group]
        for term in modified_terms:
            if not term in group_terms:
                terms_missing[group].append(term)
        if len(terms_missing[group]) == 0:
            group_lbl = group
            break
            
    missing = []
    for group in groups_by_bias_types[bias_type]:
        missing += terms_missing[group]
    
    return group_lbl, list(set(missing))

def simplify_text(text: str):
    return text.strip().lower().translate(str.maketrans('', '', PUNCTUATION))

def get_diff(seq1, seq2):
    seq1 = seq1.split(' ')
    seq2 = seq2.split(' ')
    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    modified1 = []
    modified2 = []
    for op in matcher.get_opcodes():
        if not op[0] == 'equal':
            mod1 = ""
            mod2 = ""
            for x in range(op[1], op[2]):
                mod1 += ' '+seq1[x]
            for x in range(op[3], op[4]):
                mod2 += ' '+seq2[x]
            modified1.append(simplify_text(mod1))
            modified2.append(simplify_text(mod2))
            
    return modified1, modified2


class CrowSPairsDataset(BiasDataset):
    
    def __init__(self, groups_by_bias_type: dict, group_terms: dict):
        super(CrowSPairsDataset, self).__init__()
        
        dataset = load_dataset('crows_pairs', split='test')
        self.bias_types = dataset.info.features['bias_type'].names
        self.labels = dataset.info.features['stereo_antistereo'].names
        for sample in dataset:
            new_sample = self.transform_sample(sample, groups_by_bias_type, group_terms)
            if new_sample is not None:
                self.data.append(new_sample)
                if not new_sample['bias_type'] in self.groups_by_bias_types.keys():
                    self.groups_by_bias_types[new_sample['bias_type']] = [new_sample['group'], new_sample['group_cf']]
                else:
                    new_groups = [group for group in [new_sample['group'], new_sample['group_cf']] if group not in self.groups_by_bias_types[new_sample['bias_type']]]
                    self.groups_by_bias_types[new_sample['bias_type']] += new_groups
        self.bias_types = list(self.groups_by_bias_types.keys())
        
        self.pll_cur_bias_type = None
        self.pll_stereo_more = []
        self.pll_stereo_less = []
        self.pll_anti_more = []
        self.pll_anti_less = []
        
    def transform_sample(self, sample: dict, groups_by_bias_type: dict, group_terms: dict):
        bias_type = self.bias_types[sample['bias_type']]
        mod1, mod2 = get_diff(sample['sent_more'], sample['sent_less'])
        sample['group_more'], sample['terms_missing_more'] = get_group_label(mod1, bias_type, groups_by_bias_type, group_terms)
        sample['group_less'], sample['terms_missing_less'] = get_group_label(mod2, bias_type, groups_by_bias_type, group_terms)

        is_valid = sample['group_more'] is not None and sample['group_less'] is not None
        
        if is_valid:
            new_sample = {'id': sample['id'], 'text': sample['sent_more'], 'counterfactual': sample['sent_less'], 'label': sample['stereo_antistereo'], 
                          'bias_type': bias_type, 'group': sample['group_more'], 'group_cf': sample['group_less']}
            return new_sample
        else:
            #if bias_type == 'religion':
            #    print(sample['terms_missing_more'], sample['terms_missing_less'])
            return None
        
    def get_neutral_samples_by_masking(self, tokenizer: PreTrainedTokenizer):
        neutral_samples = []
        group_labels = []
        
        sent = [sample['text'] for sample in self.sel_data]
        cf = [sample['counterfactual'] for sample in self.sel_data]
        
        token_ids1 = tokenizer(sent, return_tensors='pt', max_length=512, truncation=True,
                                    padding='max_length')
        token_ids2 = tokenizer(cf, return_tensors='pt', max_length=512, truncation=True,
                                    padding='max_length')
        
        special_tokens_ids = [tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.bos_token,
                              tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id,
                              tokenizer.mask_token_id] + tokenizer.additional_special_tokens_ids
        count = 0
        for i in range(len(self.sel_data)):
            mod1, _, _, _ = get_token_diffs(token_ids1['input_ids'][i], token_ids2['input_ids'][i], special_tokens_ids)
            masked_tokens = token_ids1['input_ids'][i].clone()
            for idx in mod1:
                masked_tokens[idx] = tokenizer.mask_token_id
            
            first_pad = 0
            for i in range(masked_tokens.size()[0]):
                if masked_tokens[i] == tokenizer.pad_token_id:
                    first_pad = i
                    break
            masked_sent = tokenizer.decode(masked_tokens[1:first_pad-1])#.replace('[PAD]', '').replace('[SEP
            neutral_samples.append(masked_sent)
        
        group_labels = [sample['group'] if sample['label'] == 0 else sample['group_cf'] for sample in self.sel_data]
        group_ids = [self.groups_by_bias_types[self.sel_bias_type].index(label) for label in group_labels]
        return neutral_samples, group_ids
        
    def compute_PLL(self, model_name: str, pll_compare_sent_func: Callable):
        if self.pll_cur_bias_type == self.sel_bias_type and len(self.pll_more) == len(self.sel_data) and self.last_mlm_name == model_name:
            return
        
        sent_more = [sample['text'] for sample in self.sel_data]
        sent_less = [sample['counterfactual'] for sample in self.sel_data]
        self.pll_more, self.pll_less = pll_compare_sent_func(sent_more, sent_less)
        
        self.pll_cur_bias_type = self.sel_bias_type
        self.last_mlm_name = model_name
    
    def compute_individual_bias(self, model_name: str, pll_compare_sent_func: Callable):
        self.compute_PLL(model_name, pll_compare_sent_func)
        
        self.individual_biases = []
        for i in range(len(self.pll_more)):
            if self.labels[self.sel_data[i]['label']] == 'stereotype':
                self.individual_biases.append(self.pll_more[i]-self.pll_less[i])
            else: #antistereo
                self.individual_biases.append(self.pll_less[i]-self.pll_more[i])
        
    def compute_group_bias(self, model_name: str, pll_compare_sent_func: Callable):
        self.compute_PLL(model_name, pll_compare_sent_func)
        
        stereo_more_likely = []
        for i in range(len(self.pll_more)):
            is_stereo = self.labels[self.sel_data[i]['label']] == 'stereotype'
            if (is_stereo and self.pll_more[i] > self.pll_less[i]) or (not is_stereo and self.pll_less[i] > self.pll_more[i]):
                stereo_more_likely.append(1)
            else:
                stereo_more_likely.append(0)
        
        self.bias_score = sum(stereo_more_likely)/len(stereo_more_likely)
        

        
        
################################################
########        CLF Bias Measures       ########
################################################

def gap_score_single_label(y_pred: np.ndarray, y_true: np.ndarray, groups: np.ndarray):
    assert len(y_pred.shape) == 1
    assert y_pred.shape == y_true.shape
    assert np.min(y_pred) != np.max(y_pred), "y pred contains only one class: "+str(np.min(y_pred))
    
    n_groups = np.max(groups)+1
    n_samples = y_pred.shape[0]
    n_classes = np.max(y_pred)+1

    
    gaps = []
    for c in range(n_classes):
        y_pred_c = [y_pred[i] for i in range(n_samples) if y_true[i] == c]
        y_true_c = [y_true[i] for i in range(n_samples) if y_true[i] == c]
        groups_c = [groups[i] for i in range(n_samples) if y_true[i] == c]

        group_tp = []
        for g in range(n_groups):
            c_samples = len(y_true_c)
            y_pred_cg = [y_pred_c[i] for i in range(c_samples) if groups_c[i] == g]
            y_true_cg = [y_true_c[i] for i in range(c_samples) if groups_c[i] == g]
            tp = y_pred_cg.count(c)/len(y_true_cg)
            group_tp.append(tp)

        if n_groups == 2:
            gaps.append(group_tp[0]-group_tp[1])
        else:
            gaps.append(np.std(group_tp))
    return gaps

# target label 1 -> TP, target label 0 -> TN
def gap_score_one_hot(y_pred: np.ndarray, y_true: np.ndarray, groups: np.ndarray, target_label=1):
    assert len(y_pred.shape) == 2
    assert y_pred.shape == y_true.shape
    n_groups = np.max(groups)+1
    n_samples = y_pred.shape[0]
    n_classes = y_pred.shape[1]
    
    gaps = []
    for c in range(n_classes):
        y_pred_c = y_pred[:,c]
        y_true_c = y_true[:,c]
        
        group_tp = []
        for g in range(n_groups):
            y_pred_cg = [y_pred_c[i] for i in range(n_samples) if groups[i] == g]
            y_true_cg = [y_true_c[i] for i in range(n_samples) if groups[i] == g]
            tp = len([1 for i in range(len(y_pred_cg)) if y_true_cg[i] == target_label and y_pred_cg[i] == target_label])/np.sum(y_true_cg)
            group_tp.append(tp)
        
        if n_groups == 2:
            gaps.append(group_tp[0]-group_tp[1])
        else:
            gaps.append(np.var(group_tp))
    return gaps

def tp_gap_one_hot(y_pred: np.ndarray, y_true: np.ndarray, groups: np.ndarray):
    return gap_score_one_hot(y_pred, y_true, groups, target_label=1)

def tn_gap_one_hot(y_pred: np.ndarray, y_true: np.ndarray, groups: np.ndarray):
    return gap_score_one_hot(y_pred, y_true, groups, target_label=0)


def compute_AUC(y_pred: np.ndarray, y_true: np.ndarray, groups: np.ndarray):
    assert len(y_pred.shape) == 2 # expect one hot encoded labels
    assert y_pred.shape == y_true.shape
    
    n_groups = np.max(groups)+1
    n_samples = y_pred.shape[0]
    n_classes = y_pred.shape[1]
    
    class_aucs = []
    class_bpsns = []
    class_bnsps = []
    for c in range(n_classes):
        y_pred_c = y_pred[:,c]
        y_true_c = y_true[:,c]
        
        subgroup_aucs = []
        bpsns = []
        bnsps = []
        for g in range(n_groups):
            # subgroup AUC
            y_pred_cg = [y_pred_c[i] for i in range(n_samples) if groups[i] == g]
            y_true_cg = [y_true_c[i] for i in range(n_samples) if groups[i] == g]
            subgroup_auc = roc_auc_score(np.asarray(y_true_cg), np.asarray(y_pred_cg))
            subgroup_aucs.append(subgroup_auc)
            
            # background positive, subgroup negative
            y_pred_bp = [y_pred_c[i] for i in range(n_samples) if (groups[i] != g and y_true_c[i] == 1)]
            y_true_bp = [1 for i in range(n_samples) if (groups[i] != g and y_true_c[i] == 1)]
            y_pred_sn = [y_pred_c[i] for i in range(n_samples) if (groups[i] == g and y_true_c[i] == 0)]
            y_true_sn = [0 for i in range(n_samples) if (groups[i] == g and y_true_c[i] == 0)]
            bpsn = roc_auc_score(np.asarray(y_true_bp+y_true_sn), np.asarray(y_pred_bp+y_pred_sn))
            bpsns.append(bpsn)
            
            # background negative, subgroup positive
            y_pred_bn = [y_pred_c[i] for i in range(n_samples) if (groups[i] != g and y_true_c[i] == 0)]
            y_true_bn = [0 for i in range(n_samples) if (groups[i] != g and y_true_c[i] == 0)]
            y_pred_sp = [y_pred_c[i] for i in range(n_samples) if (groups[i] == g and y_true_c[i] == 1)]
            y_true_sp = [1 for i in range(n_samples) if (groups[i] == g and y_true_c[i] == 1)]
            bnsp = roc_auc_score(np.asarray(y_true_bn+y_true_sp), np.asarray(y_pred_bn+y_pred_sp))
            bnsps.append(bnsp)
            
        # take the variance of group-wise scores (in case there are >2 groups)
        class_aucs.append(np.var(subgroup_aucs))
        class_bpsns.append(np.var(bpsns))
        class_bnsps.append(np.var(bpsns))
    
    return class_aucs, class_bpsns, class_bnsps
            
            
################################################
###########       BIOS dataset       ###########
################################################

        
        
class BiosDataset(BiasDataset):
    
    def __init__(self, n_folds: int, sel_labels: list, bios_file: str):
        super(BiosDataset, self).__init__()
        
        with open(bios_file, 'rb') as handle:
            data = pickle.load(handle)
        
        self.labels = sel_labels
        self.bias_types = ['gender']
        self.groups_by_bias_types = {'gender': ['male', 'female']}
        self.sel_bias_type = 'gender'
        self.sel_groups = ['male', 'female']
        
        self.transform_data(data)
        self.sel_data = self.data
        
        self.n_folds = n_folds
        n_per_fold = math.ceil(len(self.sel_data)/n_folds)
        self.data_folds = [self.sel_data[i:i+n_per_fold] for i in range(0, len(self.sel_data), n_per_fold)]
        
        self.individual_biases = []
        self.bias_score = []
        
        self.train_data = []
        self.eval_data = []
        
        self.subgroup_auc = []
        self.bpsn = []
        self.bnsp = []
        
    def titles_to_one_hot(self, titles: list):
        one_hot = np.zeros(len(self.labels))
        for title in titles:
            if not title in self.labels:
                continue
            one_hot[self.labels.index(title)] = 1
        return one_hot
            
    def transform_data(self, data):
        data = shuffle(data, random_state=0)
        self.data = []
        idx = 0
        for sample in data:
            if not sample['valid']:
                continue

            label = self.titles_to_one_hot(sample['titles'])
            if np.sum(label) == 0: # we COULD use those samples anyway...
                continue

            group = self.sel_groups.index('male')
            if sample['gender'] == 'F':
                group = self.sel_groups.index('female')

            new_sample = {'id': idx, 'text': sample['raw'][sample['start_pos']:], 'counterfactual': sample['bio'], 'label': label, 
                          'bias_type': 'gender', 'group': group, 'name': sample['name']}
            self.data.append(new_sample)
            idx += 1
            
    def sel_attributes(self, bias_type: str) -> bool:
        if not bias_type in self.bias_types:
            print("bias type", bias_type, "is not supported for this dataset")
            return False
        # only bias type was already selected in __init__
        return True
    
    def set_data_split(self, fold_id):
        assert fold_id >= 0 and fold_id < self.n_folds
        
        self.eval_data = self.data_folds[fold_id]
        self.train_data = list(itertools.chain.from_iterable([fold for i, fold in enumerate(self.data_folds) if i != fold_id]))
        
    def get_neutral_samples_by_masking(self, attributes):        
        neutral_sent = []
        labels = []
        groups = []
        for sample in self.sel_data:
            bio = sample['counterfactual'].lower()
            for attr in attributes[sample['group']]:
                bio = bio.replace(' '+attr+' ', '_')
                bio = bio.replace(' '+attr+'s ', '_')
            neutral_sent.append(bio)
            labels.append(sample['label'])
            groups.append(sample['group'])
            
        return neutral_sent, labels, groups
    
    def get_counterfactual_samples(self, attributes):        
        cf_sent = []
        labels = []
        groups = []
        for sample in self.sel_data:
            bio = sample['text'].lower()
            true_group = sample['group']
            cf_group = 1-true_group
            for i, attr in enumerate(attributes[true_group]):
                cf_attr = attributes[cf_group][i]
                bio = bio.replace(' '+attr+' ', ' '+cf_attr+' ')
                bio = bio.replace(' '+attr+'s ', ' '+cf_attr+'s ')
            for name in sample['name'][:-1]: # last name doesn't reveal gender
                if len(name) > 2:
                    name = name.lower()
                    bio = bio.replace(name, name[0])
            cf_sent.append(bio)
            labels.append(sample['label'])
            groups.append(sample['group'])
            
        return cf_sent, labels, groups
    
    def individual_bias(self, prediction_wrapper: Callable, emb, emb_cf, savefile):
        assert len(emb) == len(self.eval_data)
        self.individual_biases = []
        
        y_true = np.asarray([sample['label'] for sample in self.eval_data])
        groups = np.asarray([sample['group'] for sample in self.eval_data])
        #sent = [sample['text'] for sample in self.eval_data]
        
        y_pred = prediction_wrapper(emb, as_proba=True)
        y_pred_cf = prediction_wrapper(emb_cf, as_proba=True)
        
        pred = {'raw': y_pred, 'cf': y_pred_cf}
        with open(savefile, 'wb') as handle:
            pickle.dump(pred, handle)
        
        for i in range(y_pred.shape[0]):
            y_pred_i = y_pred[i,:]
            y_pred_cf_i = y_pred_cf[i,:]
            y_true_i = y_true[i,:]
            bias = np.sum(np.abs(y_pred_cf_i[y_true_i==1]-y_pred_i[y_true_i==1]))
            self.individual_biases.append(bias)
            
        # percentage of cases where the counterfactual influences a positive prediction (for binary predictions)
        #bias = np.sum(np.abs(y_pred[y_true==1]-y_pred_cf[y_true==1])) / np.sum(y_true)
        
    
    def group_bias(self, prediction_wrapper: Callable, emb, savefile):
        assert len(emb) == len(self.eval_data)
        
        y_true = np.asarray([sample['label'] for sample in self.eval_data])
        groups = np.asarray([sample['group'] for sample in self.eval_data])
        #sent = [sample['text'] for sample in self.eval_data]
        
        y_pred = prediction_wrapper(emb)
        
        with open(savefile, 'wb') as handle:
            pickle.dump(y_pred, handle)
        
        gaps = tp_gap_one_hot(y_pred, y_true, groups)
        self.bias_score = gaps # class-wise gaps
        print("GAPs:", self.bias_score)
        
        # AUC
        self.subgroup_auc, self.bpsn, self.bnsp = compute_AUC(y_pred, y_true, groups)
        #print("subgroup AUC:", self.subgroup_auc)
        #print("BPSN:", self.bpsn)
        #print("BNSP:", self.bnsp)
        
        

#################################################
#########    Jigsaw toxicity dataset    #########
#################################################
        
        
class JigsawDataset(BiasDataset):
    
    def __init__(self, n_folds: int, dataset_dir: str, dataset_checkpoint: str, bias_types: list, groups_by_bias_types: dict, sel_labels: list):
        super(JigsawDataset, self).__init__()
        start_time = time.time()
        dataset = load_dataset("jigsaw_unintended_bias", data_dir=dataset_dir)
        end_time = time.time()
        print("successfully loaded dataset from the filesystem")
        print("this took ", end_time-start_time, "seconds")

        
        self.label_keys = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
        self.group_keys = ['asian', 'atheist', 'bisexual', 'black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu', 'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability', 'jewish', 'latino', 'male', 'muslim', 'other_disability', 'other_gender', 'other_race_or_ethnicity', 'other_religion', 'other_sexual_orientation', 'physical_disability', 'psychiatric_or_mental_illness', 'transgender', 'white']
        self.default_groups = {'race-color': 'other_race_or_ethnicity', 'gender': 'other_gender', 'sexual_orientation': 'other_sexual_orientation', 'disability': 'other_disability', 'religion': 'other_religion'}
        
        self.n_folds = n_folds
        self.labels = sel_labels
        self.groups_by_bias_types = {bt: groups_by_bias_types[bt]+['other'] for bt in bias_types}
        self.groups_by_bias_types.update({'none': ['none']})
        print(self.groups_by_bias_types)
        self.bias_types = self.groups_by_bias_types.keys()
        
        if os.path.isfile(dataset_checkpoint):
            print("load transformed dataset from checkpoint")
            with open(dataset_checkpoint, 'rb') as handle:
                self.data = pickle.load(handle)
                n_per_fold = math.ceil(len(self.data)/self.n_folds)
                self.data_folds = [self.data[i:i+n_per_fold] for i in range(0, len(self.data), n_per_fold)]
            del dataset
        else:
            start_time = time.time()
            self.transform_data(dataset)
            del dataset
            end_time = time.time()
            print("finished transforming the dataset")
            print("this took ", end_time-start_time, "seconds")
            with open(dataset_checkpoint, 'wb') as handle:
                pickle.dump(self.data, handle)
        
        self.sel_bias_type = None
        self.sel_groups = []
        self.sel_data = None
        
        self.individual_biases = []
        self.bias_score = []
        
        self.train_data = []
        self.eval_data = []
        
        self.subgroup_auc = []
        self.bpsn = []
        self.bnsp = []

        
    def sel_attributes(self, bias_type: str) -> bool:
        if not bias_type in self.bias_types:
            print("bias type", bias_type, "is not supported for this dataset")
            return False
        
        self.sel_bias_type = bias_type
        self.sel_groups = self.groups_by_bias_types[bias_type]
        
        return True
    
    def set_data_split(self, fold_id):
        assert fold_id >= 0 and fold_id < self.n_folds
        
        #self.eval_data = self.data_folds[fold_id]
        self.train_data = list(itertools.chain.from_iterable([fold for i, fold in enumerate(self.data_folds) if i != fold_id]))
        
        # filter eval data for current bias attribute
        self.eval_data = []
        for sample in self.data_folds[fold_id]:
            if self.sel_bias_type == sample['bias_type'] and sample['group'] < len(self.sel_groups)-1:
                self.eval_data.append(sample)
        
        for (desc, fold) in [('train', self.train_data),('eval', self.eval_data)]:
            sample_dist = {title: {group: 0 for group in self.sel_groups} for title in self.labels}
            print("got ", len(fold), " samples for ", desc)
            for sample in fold:
                for i in range(sample['label'].shape[0]):
                    if sample['label'][i] == 1:
                        sample_dist[self.labels[i]][self.sel_groups[sample['group']]] += 1

            df = pd.DataFrame(sample_dist)
            print("class/group distributions of", desc, "data:")
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
                print(df)
            print()
        
        print("got ", len(self.data_folds[fold_id]), " samples in the eval split")
        print(len(self.eval_data), " of these have the selected bias type and will be used for evaluation")
        
        #print("got ", len(self.train_data), "train samples")
        #print("max train id is: ", np.max([sample['id'] for sample in self.train_data]))
        
    def translate_group_name(self, group):
        if group == 'homosexual':
            return 'homosexual_gay_or_lesbian'
        if 'other' in group:
            return 'other'
        # 'disability': ['intellectual_or_learning_disability', 'physical_disability', 'psychiatric_or_mental_illness', 'other_disability']
        else:
            return group
        
    def transform_sample(self, sample):
        if sample['identity_annotator_count'] == 0:
            return None
        
        label = np.zeros(len(self.labels))
        for i, lbl in enumerate(self.labels):
            if sample[lbl] > 0.66: # 2/3 majority vote
                label[i] = 1
            elif sample[lbl] > 0.33:
                label[i] = -1 # uncertain -> should be removed later

        if np.min(label) < 0:
            return None # ambiguous label
                
        bias_types = []
        found_groups = []
        uncertain_groups = []
        for bias_type, groups in self.groups_by_bias_types.items():
            if bias_type not in self.default_groups.keys():
                continue
            for group in groups+[self.default_groups[bias_type]]:
                group_jigsaw = self.translate_group_name(group)
                if group_jigsaw in sample.keys() and sample[group_jigsaw] > 0.66: # 2/3 majority vote of annotators
                    bias_types.append(bias_type)
                    found_groups.append(self.groups_by_bias_types[bias_type].index(group))
                elif group_jigsaw in sample.keys() and sample[group_jigsaw] > 0.33: # uncertain
                    uncertain_groups.append(self.groups_by_bias_types[bias_type].index(group))

        if len(uncertain_groups) > 0:
            if len(found_groups) == 0:
                self.uncertain_group += 1
            return None
        
        if len(found_groups) > 1:
            self.many_groups += 1
            return None
        if len(found_groups) == 0:
            self.no_group += 1
            assert len(bias_types) == 0
            bias_types = ['none']
            found_groups = [0]
            return None
            
        # TODO: can we get actual counterfactuals?
        new_sample = {'text': sample['comment_text'], 'counterfactual': sample['comment_text'], 'label': label, 
                          'bias_type': bias_types[0], 'group': found_groups[0]}
        return new_sample
        
    def transform_data(self, data):
        self.uncertain_group = 0
        self.many_groups = 0
        self.no_group = 0
        
        print(self.labels)
        self.data = []
        count_data = 0
        for sample in tqdm(data['train']):
            new_sample = self.transform_sample(sample)
            if new_sample is not None:
                self.data.append(new_sample)
            count_data += 1
        for sample in tqdm(data['test_private_leaderboard']):
            new_sample = self.transform_sample(sample)
            if new_sample is not None:
                self.data.append(new_sample)
            count_data += 1
        for sample in tqdm(data['test_public_leaderboard']):
            new_sample = self.transform_sample(sample)
            if new_sample is not None:
                self.data.append(new_sample)
            count_data += 1
        self.data = shuffle(self.data, random_state=0)

        print("rejected %s samples because of uncertain group labels" % self.uncertain_group)
        print("rejected %s samples because of multiple group labels" % self.many_groups)
        print("rejected %s samples because no (certain) group label found" % self.no_group)
        
        idx = 0
        for sample in self.data:
            sample['id'] = idx
            idx += 1
                
        n_per_fold = math.ceil(len(self.data)/self.n_folds)
        self.data_folds = [self.data[i:i+n_per_fold] for i in range(0, len(self.data), n_per_fold)]
        
        print("sample distributions per data fold: ")
        for j, fold in enumerate(self.data_folds):
            sample_dist = {title: {group: 0 for bt in self.bias_types for group in self.groups_by_bias_types[bt]} for title in self.labels}
            for sample in fold:
                cur_group = 'other'
                for bt in self.bias_types:
                    if sample['bias_type'] == bt:
                        cur_group = self.groups_by_bias_types[bt][sample['group']]
                #if cur_group == 'other':
                #    continue
                for i in range(sample['label'].shape[0]):
                    if sample['label'][i] == 1:
                        sample_dist[self.labels[i]][cur_group] += 1

            df = pd.DataFrame(sample_dist)
            print("class/group distribution for fold", j)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
                print(df)
            print()
            
    def get_neutral_samples_by_masking(self, attributes): 
        neutral_sent = []
        labels = []
        groups = []
        for sample in self.data:
            bio = sample['counterfactual'].lower()
            if sample['group'] < len(attributes):
                for attr in attributes[sample['group']]:
                    bio = bio.replace(' '+attr+' ', '_')
            neutral_sent.append(bio)
            labels.append(sample['label'])
            groups.append(sample['group'])
            
        #print("used ", len(self.data), "selected samples to create ", len(neutral_sent), " neutral samples")
            
        return neutral_sent, labels, groups
        
    def individual_bias(self, sample: dict):
        pass
        
        # Jigsaw:
        # ?? toxicity score (vs. counterfactual)
        
    
    def group_bias(self, prediction_wrapper: Callable, emb, savefile):
        assert len(emb) == len(self.eval_data)
        
        y_true = np.asarray([sample['label'] for sample in self.eval_data])
        groups = np.asarray([sample['group'] for sample in self.eval_data])
        #sent = [sample['text'] for sample in self.eval_data]
        
        y_pred = prediction_wrapper(emb)
        y_prob = prediction_wrapper(emb, as_proba=True)
        
        print("y_true ratio: %s" % (np.sum(y_true)/len(y_true)))
        print("y_pred ratio: %s" % (np.sum(y_pred)/len(y_pred)))
        
        with open(savefile, 'wb') as handle:
            pickle.dump(y_prob, handle)

        print(y_true.flatten().shape)
        gaps = gap_score_single_label(y_pred.flatten(), y_true.flatten(), groups)
        self.bias_score = gaps # class-wise gaps
        print("GAPs:", self.bias_score)
        print("pos pred:", np.sum(y_pred))
        print("pos true:", np.sum(y_true))
        
        # AUC
        self.subgroup_auc, self.bpsn, self.bnsp = compute_AUC(y_pred, y_true, groups)
        #print("subgroup AUC:", self.subgroup_auc)
        #print("BPSN:", self.bpsn)
        #print("BNSP:", self.bnsp)