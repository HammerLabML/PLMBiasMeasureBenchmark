import difflib
import string

import datasets
from datasets import load_dataset

from embedding import BertHuggingfaceMLM
from transformers import PreTrainedTokenizer

from unmasking_bias import PLLBias, get_token_diffs

from sklearn.utils import shuffle
import itertools
from collections.abc import Callable
import numpy as np
import math
import pickle

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
        
    def compute_PLL(self, mlm: BertHuggingfaceMLM):
        if self.pll_cur_bias_type == self.sel_bias_type and len(self.pll_more) == len(self.sel_data) and self.last_mlm_name == mlm.model.config._name_or_path:
            return
        pllBias = PLLBias(mlm.model, mlm.tokenizer, mlm.batch_size)
        
        sent_more = [sample['text'] for sample in self.sel_data]
        sent_less = [sample['counterfactual'] for sample in self.sel_data]
        self.pll_more, self.pll_less = pllBias.compare_sentence_likelihood(sent_more, sent_less)
        
        self.pll_cur_bias_type = self.sel_bias_type
        self.last_mlm_name = mlm.model.config._name_or_path
    
    def compute_individual_bias(self, mlm: BertHuggingfaceMLM):
        self.compute_PLL(mlm)
        
        self.individual_biases = []
        for i in range(len(self.pll_more)):
            if self.labels[self.sel_data[i]['label']] == 'stereotype':
                self.individual_biases.append(self.pll_more[i]-self.pll_less[i])
            else: #antistereo
                self.individual_biases.append(self.pll_less[i]-self.pll_more[i])
        
    def compute_group_bias(self, mlm: BertHuggingfaceMLM):
        self.compute_PLL(mlm)
        
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

def gap_score_single_label(y_pred: np.ndarray, y_true: np.ndarray, groups: list):
    assert len(y_pred.shape) == 1
    
    n_groups = max(groups)+1
    n_samples = y_pred.shape[0]
    n_classes = max(y_pred)+1
    
    gaps = []
    for c in range(n_classes):
        y_pred_c = [y_pred[i] for i in range(n_samples) if y_true == c]
        y_true_c = [y_true[i] for i in range(n_samples) if y_true == c]
        groups_c = [groups[i] for i in range(n_samples) if y_true == c]
        
        group_tp = []
        for g in range(n_groups):
            y_pred_cg = [y_pred_c[i] for i in range(n_samples) if groups[i] == g]
            y_true_cg = [y_true_c[i] for i in range(n_samples) if groups[i] == g]
            tp = y_pred_cg.count(c)/len(y_true_cg)
            group_tp.append(tp)
        
        if n_groups == 2:
            gaps.append(group_tp[0]-group_tp[1])
        else:
            gaps.append(np.var(group_tp))
    return gaps

def gap_score_one_hot(y_pred: np.ndarray, y_true: np.ndarray, groups: list):
    assert len(y_pred.shape) == 2
    n_groups = max(groups)+1
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
            tp = len([1 for i in range(len(y_pred_cg)) if y_true_cg[i] == 1 and y_pred_cg[i] == 1])/np.sum(y_true_cg)
            group_tp.append(tp)
        
        if n_groups == 2:
            gaps.append(group_tp[0]-group_tp[1])
        else:
            gaps.append(np.var(group_tp))
    return gaps


# TODO: ROC AUC based metrics

        
        
################################################
###########       BIOS dataset       ###########
################################################

        
        
class BiosDataset(BiasDataset):
    
    def __init__(self, n_folds: int, sel_labels: list, bios_file: str):
        super(BiosDataset, self).__init__()
        
        with open(bios_file, 'rb') as handle:
            data = pickle.load(handle)
        
        # all titles with >100 samples
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
        
    def titles_to_one_hot(self, titles: list):
        one_hot = np.zeros(len(self.labels))
        for title in titles:
            if not title in self.labels:
                continue
            one_hot[self.labels.index(title)] = 1
        return one_hot
            
    def transform_data(self, data):
        self.data = []
        for sample in data:
            if not sample['valid']:
                continue

            label = self.titles_to_one_hot(sample['titles'])
            if np.sum(label) == 0: # we COULD use those samples anyway...
                continue

            group = self.sel_groups.index('male')
            if sample['gender'] == 'F':
                group = self.sel_groups.index('female')

            new_sample = {'id': sample['Unnamed: 0'], 'text': sample['raw'][sample['start_pos']:], 'counterfactual': sample['bio'], 'label': label, 
                          'bias_type': 'gender', 'group': group}
            self.data.append(new_sample)
        self.data = shuffle(self.data, random_state=0)
            
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
        
    def get_neutral_samples_by_masking(self, attributes, split='train'):
        if split == 'train':
            data = self.train_data
        elif split == 'eval':
            data = self.eval_data
        else:
            return []
        
        neutral_sent = []
        labels = []
        groups = []
        for sample in data:
            bio = sample['counterfactual'].lower()
            for attr in attributes[sample['group']]:
                bio = bio.replace(' '+attr+' ', '_')
            neutral_sent.append(bio)
            labels.append(sample['label'])
            groups.append(sample['group'])
            
        return neutral_sent, labels, groups
    
    def individual_bias(self, sample: dict):
        pass
        
        # BIOS:
        # ?? counterfactual prediction diff/ bool does prediction change
    
    def group_bias(self, prediction_wrapper: Callable):
        pass
        
        y_true = [sample['label'] for sample in self.eval_data]
        groups = [sample['group'] for sample in self.eval_data]
        sent = [sample['text'] for sample in self.eval_data]
        
        y_pred = prediction_wrapper(sent)
        
        gaps = gap_score_one_hot(y_pred, np.asarray(y_true), groups)
        self.bias_score = gaps # class-wise gaps
        
    def roc_auc_bias(self):
        pass
        #TODO
        
        
        

#################################################
#########    Jigsaw toxicity dataset    #########
#################################################
        
        
class JigsawDataset(BiasDataset):
    
    def __init__(self):
        super(CrowSPairsDataset, self).__init__()
    
    def individual_bias(self, sample: dict):
        pass
        
        # Jigsaw:
        # ?? toxicity score (vs. counterfactual)
        
    
    def group_bias(self):
        pass
        # maybe also GAP-like score?
        
    def roc_auc_bias(self):
        pass
        #TODO