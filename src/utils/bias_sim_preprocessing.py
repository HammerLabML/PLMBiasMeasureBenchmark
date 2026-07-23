import random
import pandas as pd
import numpy as np
import yaml
import math
import torch
from transformers import AutoModelForMaskedLM, PreTrainedTokenizer
from unmasking_bias import PLLBias, get_token_diffs, get_modified_tokens_from_sent
from typing import List
from .mlm import apply_random_masking


def create_bias_distribution(n_groups: int, target_words: list, minP: float = 0.0, maxP: float = 1.0):
    """
    Generates a dictionary of probability distributions for multiple target words to co-occur with demographic groups.
    
    For each target word and attribute (e.g. gender, ethnicity), this function randomly assigns probabilities to n_groups such that:
    - The sum of probabilities equals 1.0.
    - Each group has a probability strictly between minP and maxP (with adjustments to ensure the last group fits).
    - The distribution is uniform-randomly generated but constrained by the bounds.

    Args:
        n_groups (int): The number of demographic groups to distribute probabilities among.
        target_words (list): A list of target terms for which distributions will be created.
        minP (float, optional): Minimum probability allowed for any single group. Must be < 1.0 / n_groups.
                                Defaults to 0.0.
        maxP (float, optional): Maximum probability allowed for any single group. Defaults to 1.0.

    Returns:
        dict: A nested dictionary where the outer key is a target word, and the value is another dictionary mapping
              group IDs (int) to their assigned probability (float).
              Format: {target_word: {group_id: probability, ...}, ...}

    Raises:
        AssertionError: If minP condition (< 1 / n_groups) is not met.

    Example:
        dist = create_bias_distribution(3, ["doctor", "Bnurseob"], minP=0.1, maxP=0.6)
        # Possible output: {'doctor': {0: 0.2, 1: 0.5, 2: 0.3}, 'nurse': {0: 0.4, 1: 0.4, 2: 0.2}}
    """

    assert minP < 1.0 / n_groups, "minP must be in [0, 1/n_groups)"

    probs_by_target = {}
    for target in target_words:
        probs_by_target.update({target: {}})

    for target in target_words:
        P = 1.0
        groups = list(range(n_groups))
        while len(groups) > 1:
            i = random.choice(groups)
            groups.remove(i)
            p = random.uniform(minP, min(maxP, P - minP * len(groups)))
            P -= p
            probs_by_target[target].update({i: p})

        # last group
        i = groups[0]
        p = P
        probs_by_target[target].update({i: p})

    return probs_by_target


def mask_by_ids(token_ids: torch.Tensor, to_mask_ids: list, mask_token_id) -> torch.Tensor:
    """
    Replaces specific token IDs in a tensor with the mask token ID.

    Args:
        token_ids (torch.Tensor): The input tensor of token IDs (shape expected to be (1, seq_len)).
        to_mask_ids (list): A list of integer indices (positions in the sequence) to mask.
        mask_token_id (int): The ID to assign to the masked positions.

    Returns:
        torch.Tensor: A new tensor identical to `token_ids` but with the specified positions set to `mask_token_id`.
    """
    masked_tokens = token_ids.clone()
    for idx in to_mask_ids:
        masked_tokens[0][idx] = mask_token_id
    return masked_tokens


def replace_attribute(sentence: str, template_config: dict, protected_attribute: str, group_id=0, neutral=False, mask=False):
    """
    Replaces a protected attribute term in a sentence with a term from a specific group, a neutral term, or a mask.
    
    This function scans the sentence for keys defined in `template_config` for the given `protected_attribute`.
    When a key is found, it is replaced according to the flags:
    - `neutral`: Replaces with the first term in the template list (index 0).
    - `mask`: Replaces with '[MASK]'.
    - Else: Replaces with the term corresponding to `group_id` (offset by +1 since index 0 is usually neutral).

    Note: This is intended for the evaluation templates where exactly one key per attribute is expected. If multiple keys were given,
          only the last key and replace term would be returned.

    Args:
        sentence (str): The input text string to modify.
        template_config (dict): Configuration dictionary containing attribute mappings. Expected structure:
                                {attribute_name: {"KEYS": [key1, key2], key1: [neutral_term, group1_term, group2_term...]}}.
        protected_attribute (str): The key in `template_config` corresponding to the attribute to replace.
        group_id (int, optional): The index of the group to use for replacement (0-based relative to non-neutral terms).
                                  Defaults to 0.
        neutral (bool, optional): If True, replaces with the neutral term (index 0). Mutually exclusive with `mask`.
                                  Defaults to False.
        mask (bool, optional): If True, replaces with '[MASK]'. Mutually exclusive with `neutral`.
                               Defaults to False.

    Returns:
        tuple: A tuple containing:
            - `sentence` (str): The sentence with the replacement applied.
            - `replaced_term` (str): The term that was inserted into the sentence.
            - `matched_key` (str or None): The original key in the sentence that was matched and replaced.

    Raises:
        AssertionError: If both `neutral` and `mask` are set to True.
    """
    assert not (neutral and mask), "both neutral and mask were set true, but only one can apply!"

    replaced_term = ''
    matched_key = None

    keys = template_config[protected_attribute]['KEYS']
    for key in keys:
        if key in sentence:
            if neutral:
                replaced_term = template_config[key][0]
            elif mask:
                replaced_term = '[MASK]'
            else:
                replaced_term = template_config[key][group_id+1]  # offset for neutral term
            sentence = sentence.replace(key, replaced_term)
            if matched_key is not None:
                print("warning: got multiple keys for one protected attribute")
            matched_key = key

    # sentence after attr replacement; that that was inserted; attribute key that was replaced
    return sentence, replaced_term, matched_key


def attr_in_template(template, protected_attr, template_config):
    """
    Checks if any of the predefined keys for a protected attribute exist in a template string.

    Args:
        template (str): The string (sentence template) to search in.
        protected_attr (str): The name of the protected attribute to find in `template_config`.
        template_config (dict): The configuration dictionary containing the 'KEYS' list for the attribute.

    Returns:
        bool: True if at least one key associated with `protected_attr` is found in `template`, False otherwise.
    """
    keys = template_config[protected_attr]['KEYS']

    found_attr = False
    for key in keys:
        if key in template:
            found_attr = True
        
    return found_attr


def templates_to_eval_samples(tokenizer: PreTrainedTokenizer, template_config: dict, target_words: list, template_key: str):
    data = []
    data_prior = [] # with masked out occupations for group prior
    mask_str = 'person' # using mask token leads to problems so just neutral term person instead

    # these special tokens should be ignored
    special_tokens_ids = [tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.bos_token,
                          tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id,
                          tokenizer.mask_token_id] + tokenizer.additional_special_tokens_ids

    for temp in template_config[template_key]:
        for target in target_words+[mask_str]:
            sentence_base = temp.replace(template_config['target'], target)
            sentence_attr_base_no_target = temp

            for protected_attr in template_config['protected_attr']:
                groups = template_config[protected_attr]['GROUPS'][1:]
                keys = template_config[protected_attr]['KEYS']

                if not attr_in_template(temp, protected_attr, template_config):
                    continue

                entry = {'template': temp, 'target': target, 'sentences': None, 'sent_masked_attr': None, 'attr_key': '', 'protected_attr': '',
                     'attr_choices': None, 'attribute_token_ids': None, 'non_attr_token_ids': None, 'target_token_ids': None}

                # replace any other protected attribute with neutral terms:
                sentence_attr_base = sentence_base
                sentence_attr_base_no_target = temp
                for other_pattr in template_config['protected_attr']:
                    if other_pattr == protected_attr or not attr_in_template(temp, other_pattr, template_config):
                        continue

                    sentence_attr_base, _, _ = replace_attribute(sentence_attr_base, template_config, other_pattr, neutral=True)
                    sentence_attr_base_no_target, _, _ = replace_attribute(sentence_attr_base_no_target, template_config, other_pattr, neutral=True)

                # create one sample per group of the chosen attribute
                sentences = []
                sentences_no_target = []
                terms = []
                for k, group in enumerate(groups):
                    sent, term, _ = replace_attribute(sentence_attr_base, template_config, protected_attr, group_id=k)
                    sentences.append(sent)
                    terms.append(term)
                    sent_nt, _, _ = replace_attribute(sentence_attr_base_no_target, template_config,
                                                                 protected_attr, group_id=k)
                    sentences_no_target.append(sent_nt)

                entry['sent_masked_attr'], _, entry['attr_key'] = replace_attribute(sentence_attr_base, template_config, protected_attr, mask=True)
                entry['sentences'] = tuple(sentences)
                entry['protected_attr'] = protected_attr
                entry['attr_choices'] = terms
                # token ids for sentences that differ only by attributes
                token_ids = tokenizer(sentences, return_tensors='pt', max_length=512, truncation=True,
                                      padding='max_length')

                # compare tokenized sentences pairwise to get attribute/ non-attribute token ids
                n_versions = len(sentences)
                attr_ids = []
                non_attr_ids = []
                for i in range(0, n_versions, 2):
                    if i == n_versions-1:
                        # compare with first sentence
                        attr_ids1, _, non_attr_ids1, _ = get_token_diffs(token_ids['input_ids'][i],
                                                                         token_ids['input_ids'][0], special_tokens_ids)
                        attr_ids.append(attr_ids1)
                        non_attr_ids.append(non_attr_ids1)
                    else:
                        # compare the two next sentences
                        attr_ids1, attr_ids2, non_attr_ids1, non_attr_ids2 = get_token_diffs(token_ids['input_ids'][i],
                                                                                             token_ids['input_ids'][i+1],
                                                                                             special_tokens_ids)

                        attr_ids.append(attr_ids1)
                        attr_ids.append(attr_ids2)
                        non_attr_ids.append(non_attr_ids1)
                        non_attr_ids.append(non_attr_ids2)

                entry['attribute_token_ids'] = tuple(attr_ids)
                entry['non_attr_token_ids'] = tuple(non_attr_ids)

                #  determine the target
                token_ids_no_target = tokenizer(sentences_no_target, return_tensors='pt', max_length=512,
                                                truncation=True, padding='max_length')
                target_ids = []
                for i in range(0, n_versions):
                    target_ids1, _, _, _ = get_token_diffs(token_ids['input_ids'][i],
                                                           token_ids_no_target['input_ids'][i], special_tokens_ids)
                    target_ids.append(target_ids1)
                entry['target_token_ids'] = tuple(target_ids)

                assert entry['sentences'] is not None, "could not generate test sentences for template: "+temp

                if target == mask_str:
                    data_prior.append(entry)
                else:
                    data.append(entry)

    return data, data_prior


def templates_to_train_samples(tokenizer: PreTrainedTokenizer, template_config: dict, probs_by_attr: dict,
                               target_words: list, config: dict, template_key: str):
    masking_strategy = config['masking_strategy']
    mask_prob = config['mask_prob']
    data = []

    special_tokens_ids = [tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.bos_token,
                          tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id,
                          tokenizer.mask_token_id] + tokenizer.additional_special_tokens_ids

    for temp in template_config[template_key]:
        found_attr = False
        for target in target_words:
            sentence = temp.replace(template_config['target'], target)
            sentence_attr_base = sentence

            entry = {'template': temp, 'target': target, 'sentence': '', 'masked_sentence': '',
                     'attribute_token_ids': [], 'non_attr_token_ids': [], 'target_token_ids': []}

            for protected_attr in template_config['protected_attr']:
                groups = template_config[protected_attr]['GROUPS'][1:]
                keys = template_config[protected_attr]['KEYS']

                if not attr_in_template(temp, protected_attr, template_config):
                    entry[protected_attr] = -1
                    continue

                found_attr = True

                # derive the protected group based on group-target probabilities
                probs = probs_by_attr[protected_attr][target]
                k = 0
                r = random.uniform(0.0, 1.0)
                p = 0
                for i in range(len(probs)):
                    p += probs[i]
                    if r < p:
                        k = i
                        break
                entry[protected_attr] = k

                for key in keys:
                    if key in temp:
                        sentence = sentence.replace(key, template_config[key][k+1]) # index 0 is neutral and ignored here


            if not found_attr:
                print("could not replace attributes in: ", sentence)
                continue

            # now all attributes have been replaced
            # determine modified/ unmodified token ids
            token_ids = tokenizer(sentence, return_tensors='pt', truncation=True)
            token_attr_diff = tokenizer(sentence_attr_base, return_tensors='pt', truncation=True)
            sentence_target_base = sentence.replace(target, template_config['target'])
            token_target_diff = tokenizer(sentence_target_base, return_tensors='pt', truncation=True)

            mod_attr, _, unmod_attr, _ = get_token_diffs(token_ids['input_ids'][0], token_attr_diff['input_ids'][0],
                                                         special_tokens_ids)
            mod_target, _, _, _ = get_token_diffs(token_ids['input_ids'][0], token_target_diff['input_ids'][0],
                                                  special_tokens_ids)
            entry['attribute_token_ids'] = mod_attr
            entry['non_attr_token_ids'] = unmod_attr
            entry['target_token_ids'] = mod_target
            entry['sentence'] = sentence  # label for unmasking (y)

            special_tokens_mask = token_ids.get('special_tokens_mask', None)
            if special_tokens_mask is None:
                special_tokens_mask = torch.tensor([
                    tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True) 
                    for ids in token_ids['input_ids'].cpu().tolist()
                ], device='cpu')

            attention_mask = token_ids['attention_mask']
            candidate_mask = (~special_tokens_mask.bool()) & (attention_mask.bool())

            # Option1: mask all attribute tokens
            if masking_strategy == 'attribute':
                masked_token_ids = mask_by_ids(token_ids['input_ids'], to_mask_ids=mod_attr,
                                               mask_token_id=tokenizer.mask_token_id)

            # Option2: random masking on non-attribute tokens
            elif masking_strategy == 'non_attribute':
                filtered_mask = torch.zeros_like(candidate_mask)
                flat_valid_ids = torch.tensor(unmod_attr, device='cpu')
                filtered_mask.flatten()[flat_valid_ids] = 1
                #filtered_mask[unmod_attr] = 1
                candidate_mask = (candidate_mask) & (filtered_mask)
                masked_token_ids, _ = apply_random_masking(token_ids['input_ids'], tokenizer.mask_token_id, len(tokenizer), 
                                                        token_mask=candidate_mask, mask_prob=mask_prob)

            # Option3: mask all target tokens
            elif masking_strategy == 'target':
                masked_token_ids = mask_by_ids(token_ids['input_ids'], to_mask_ids=mod_target,
                                               mask_token_id=tokenizer.mask_token_id)
            # Option4: random masking
            else:  # masking_strategy == 'random'
                masked_token_ids, _ = apply_random_masking(token_ids['input_ids'], tokenizer.mask_token_id, len(tokenizer), 
                                                        token_mask=candidate_mask, mask_prob=mask_prob)

            masked_sentence = tokenizer.decode(masked_token_ids[0][1:masked_token_ids.size()[1]-1])
            entry['masked_sentence'] = masked_sentence  # masked sample (X)

            if not '[MASK]' in masked_sentence:
                print("found nothing to mask in: ", sentence)
                continue
            
            data.append(entry)

    return data


def create_masked_dataset(template_config, probs_by_attr, target_words, template_key='templates_train'):
    X = []  # masked sentences
    y = []  # complete sentence
    data = []

    n_templates = len(template_config[template_key])
    for temp in template_config[template_key][:n_templates]:
        for target in target_words:
            sentence = temp.replace(template_config['target'], target)

            entry = {'template': temp, 'target': target, 'sentence': '', 'masked_sentences': []}
            replace_terms = []  # terms by which the tokens are replaced

            for protected_attr in template_config['protected_attr']:
                groups = template_config[protected_attr]['GROUPS'][1:]
                keys = template_config[protected_attr]['KEYS']

                if not attr_in_template(temp, protected_attr, template_config):
                    entry.update({protected_attr: -1})
                    continue

                # derive the protected group based on group-target probabilities
                probs = probs_by_attr[protected_attr][target]
                k = 0
                r = random.uniform(0.0, 1.0)
                p = 0
                for i in range(len(probs)):
                    p += probs[i]
                    if r < p:
                        k = i
                        break
                entry.update({protected_attr: k})

                for key in keys:
                    if key in temp:
                        sentence = sentence.replace(key, template_config[key][k+1]) # index 0 is neutral and ignored here
                        replace_terms.append(template_config[key][k+1])
        
            for i, term in enumerate(replace_terms):
                masked = sentence.replace(term, '[MASK]')
                entry['masked_sentences'].append(masked)
            entry['sentence'] = sentence
            data.append(entry)

    for sample in data:
        for mask in sample['masked_sentences']:
            X.append(mask)
            y.append(sample['sentence'])

    return data, X, y