import random
import pandas as pd
import numpy as np
import yaml
import math
import torch
from transformers import AutoModelForMaskedLM, PreTrainedTokenizer
from unmasking_bias import PLLBias, get_token_diffs, get_modified_tokens_from_sent
from typing import List


def create_bias_distribution(n_groups: int, target_words: list, minP: float = 0.0, maxP: float = 1.0):
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


def random_masking(token_ids: torch.Tensor, mask_token_id, ignore_token_ids=None, token_id_subset=None,
                   mask_prob: float = 0.15) -> torch.Tensor:
    if token_id_subset is None:
        token_id_subset = list(range(token_ids.size()[1]))
    else:
        assert type(token_id_subset) == list, "expected a list of token ids or None"

    # don't replace any special tokens
    final_token_subset = []
    if ignore_token_ids is not None:
        for idx in token_id_subset:
            if int(token_ids[0][idx]) not in ignore_token_ids:
                final_token_subset.append(idx)
    else:
        final_token_subset = token_id_subset

    n = len(final_token_subset)
    np.random.shuffle(final_token_subset)
    to_mask_ids = final_token_subset[:math.ceil(mask_prob * n)]

    assert len(to_mask_ids) > 0, "need to replace at least 1 token with a [MASK]"

    return mask_by_ids(token_ids, to_mask_ids, mask_token_id)


def mask_by_ids(token_ids: torch.Tensor, to_mask_ids: list, mask_token_id) -> torch.Tensor:
    masked_tokens = token_ids.clone()
    for idx in to_mask_ids:
        masked_tokens[0][idx] = mask_token_id
    return masked_tokens


def replace_attribute(sentence: str, template_config: dict, protected_attribute: str, group_id=0, neutral=False):
    for i in range(len(template_config[protected_attribute]) - 1, -1, -1):
        cur_attr = protected_attribute + str(i)
        if cur_attr in sentence:
            if neutral:
                sentence = sentence.replace(cur_attr, template_config[protected_attribute + '_neutral'][i])
            else:
                sentence = sentence.replace(cur_attr, template_config[protected_attribute][i][group_id])
    return sentence


def templates_to_eval_samples(tokenizer: PreTrainedTokenizer, template_config: dict, target_words: list):
    data = []

    # these special tokens should be ignored
    special_tokens_ids = [tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.bos_token,
                          tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id,
                          tokenizer.mask_token_id] + tokenizer.additional_special_tokens_ids

    for temp in template_config['templates_test']:
        for target in target_words:
            sentence_base = temp.replace(template_config['target'], target)
            sentence_attr_base_no_target = temp

            entry = {'template': temp, 'target': target, 'sentences': None, 'protected_attr': '',
                     'attribute_token_ids': None, 'non_attr_token_ids': None, 'target_token_ids': None}

            for protected_attr in template_config['protected_attr']:
                if protected_attr not in temp:
                    continue

                # replace any other protected attribute with neutral terms:
                sentence_attr_base = sentence_base
                sentence_attr_base_no_target = temp
                for other_pattr in template_config['protected_attr']:
                    if other_pattr == protected_attr or other_pattr not in temp:
                        continue

                    sentence_attr_base = replace_attribute(sentence_attr_base, template_config, other_pattr, neutral=True)
                    sentence_attr_base_no_target = replace_attribute(sentence_attr_base_no_target, template_config, other_pattr, neutral=True)

                # create one sample per group of the chosen attribute
                sentences = []
                sentences_no_target = []
                for k, group in enumerate(template_config[protected_attr][0]):
                    sentences.append(replace_attribute(sentence_attr_base, template_config, protected_attr, group_id=k))
                    sentences_no_target.append(replace_attribute(sentence_attr_base_no_target, template_config,
                                                                 protected_attr, group_id=k))

                entry['sentences'] = tuple(sentences)
                entry['protected_attr'] = protected_attr

                # token ids for sentences that differ only by attributes
                token_ids = tokenizer(sentences, return_tensors='pt', truncation=True)

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
                token_ids_no_target = tokenizer(sentences_no_target, return_tensors='pt', truncation=True)
                target_ids = []
                for i in range(0, n_versions):
                    target_ids1, _, _, _ = get_token_diffs(token_ids['input_ids'][i],
                                                           token_ids_no_target['input_ids'][i], special_tokens_ids)
                    target_ids.append(target_ids1)
                entry['target_token_ids'] = tuple(target_ids)

            assert entry['sentences'] is not None, "could not generate test sentences for template: "+temp
            data.append(entry)

    return data


def templates_to_train_samples(tokenizer: PreTrainedTokenizer, template_config: dict, probs_by_attr: dict,
                               target_words: list, config: dict):
    masking_strategy = config['masking_strategy']
    mask_prob = config['mask_prob']
    data = []

    special_tokens_ids = [tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.bos_token,
                          tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id,
                          tokenizer.mask_token_id] + tokenizer.additional_special_tokens_ids

    for temp in template_config['templates_train']:
        for target in target_words:
            sentence = temp.replace(template_config['target'], target)
            sentence_attr_base = sentence

            entry = {'template': temp, 'target': target, 'sentence': '', 'masked_sentence': '',
                     'attribute_token_ids': [], 'non_attr_token_ids': [], 'target_token_ids': []}

            for protected_attr in template_config['protected_attr']:
                if protected_attr not in temp:
                    entry[protected_attr] = -1
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
                entry[protected_attr] = k

                # go backward so that GENDER11 doesn't get confused with GENDER1
                for i in range(len(template_config[protected_attr]) - 1, -1, -1):
                    cur_attr = protected_attr + str(i)
                    if cur_attr in temp:
                        sentence = sentence.replace(cur_attr, template_config[protected_attr][i][k])

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

            # Option1: mask all attribute tokens
            if masking_strategy == 'attribute':
                masked_token_ids = mask_by_ids(token_ids['input_ids'], to_mask_ids=mod_attr,
                                               mask_token_id=tokenizer.mask_token_id)

            # Option2: random masking on non-attribute tokens
            elif masking_strategy == 'non_attribute':
                masked_token_ids = random_masking(token_ids['input_ids'], tokenizer.mask_token_id,
                                                  ignore_token_ids=special_tokens_ids, token_id_subset=unmod_attr,
                                                  mask_prob=mask_prob)

            # Option3: mask all target tokens
            elif masking_strategy == 'target':
                masked_token_ids = mask_by_ids(token_ids['input_ids'], to_mask_ids=mod_target,
                                               mask_token_id=tokenizer.mask_token_id)
            # Option4: random masking
            else:  # masking_strategy == 'random'
                masked_token_ids = random_masking(token_ids['input_ids'], tokenizer.mask_token_id,
                                                  ignore_token_ids=special_tokens_ids, token_id_subset=None,
                                                  mask_prob=mask_prob)

            masked_sentence = tokenizer.decode(masked_token_ids[0][1:masked_token_ids.size()[1]-1])
            entry['masked_sentence'] = masked_sentence  # masked sample (X)
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
                if not protected_attr in temp:
                    entry.update({protected_attr: -1})
                    continue

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

                # go backward so that GENDER11 doesn't get confused with GENDER1
                for i in range(len(template_config[protected_attr]) - 1, -1, -1):
                    cur_attr = protected_attr + str(i)
                    if cur_attr in temp:
                        replace_terms.append(template_config[protected_attr][i][k])
                        sentence = sentence.replace(cur_attr, template_config[protected_attr][i][k])

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