import difflib
import string

import datasets
from datasets import load_dataset

from embedding import BertHuggingfaceMLM
from transformers import PreTrainedTokenizer

from unmasking_bias import PLLBias, get_token_diffs

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
        
        #stereo_more = [sample['text'] for sample in self.sel_data if sample['label'] == self.labels.index('stereo')]
        #stereo_less = [sample['counterfactual'] for sample in self.sel_data if sample['label'] == self.labels.index('stereo')]
        #antistereo_more = [sample['text'] for sample in self.sel_data if sample['label'] == self.labels.index('antistereo')]
        #antistereo_less = [sample['counterfactual'] for sample in self.sel_data if sample['label'] == self.labels.index('antistereo')]
        #self.pll_stereo_more, self.pll_stereo_less = pllBias.compare_sentence_likelihood(stereo_more, stereo_less)
        #self.pll_anti_more, self.pll_anti_less = pllBias.compare_sentence_likelihood(antistereo_more, antistereo_less)
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
                
#        for i in range(len(self.pll_stereo_more)):
#            self.individual_biases.append(self.pll_stereo_more[i]-self.pll_stereo_less[i])
#        for i in range(len(self.pll_anti_more)):
#            self.individual_biases.append(self.pll_anti_less[i]-self.pll_anti_more[i])
        
    def compute_group_bias(self, mlm: BertHuggingfaceMLM):
        self.compute_PLL(mlm)
        
        stereo_more_likely = []
        for i in range(len(self.pll_more)):
            is_stereo = self.labels[self.sel_data[i]['label']] == 'stereotype'
            if (is_stereo and self.pll_more[i] > self.pll_less[i]) or (not is_stereo and self.pll_less[i] > self.pll_more[i]):
                stereo_more_likely.append(1)
            else:
                stereo_more_likely.append(0)
        #stereo_more_likely = [1 if self.pll_stereo_more[i] > self.pll_stereo_less[i] else 0 for i in range(len(self.pll_stereo_less))]
        #stereo_more_likely += [1 if self.pll_anti_less[i] > self.pll_anti_more[i] else 0 for i in range(len(self.pll_anti_more))]
        
        self.bias_score = sum(stereo_more_likely)/len(stereo_more_likely)
        
        
class BiosDataset(BiasDataset):
    
    def __init__(self):
        super(CrowSPairsDataset, self).__init__()
    
    def individual_bias(self, sample: dict):
        pass
        
        # BIOS:
        # ?? counterfactual prediction diff/ bool does prediction change
        
    
    def group_bias(self):
        pass
        
        # BIOS (clf):
        # GAP (diff) or variance of class-wise probabilities over protected groups
        # maybe AUC
        
        
class JigsawDataset(BiasDataset):
    
    def __init__(self):
        super(CrowSPairsDataset, self).__init__()
    
    def individual_bias(self, sample: dict):
        pass
        
        # Jigsaw:
        # ?? toxicity score (vs. counterfactual)
        
    
    def group_bias(self):
        pass
        
        # Jigsaw (regression):
        # subgroup AUC (vs. bias AUC)
        # maybe also GAP-like score?