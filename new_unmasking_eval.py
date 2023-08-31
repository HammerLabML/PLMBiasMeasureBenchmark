import numpy as np
import os
import pandas as pd
import pickle
import scipy
import yaml
import getopt
import sys

import torch
from transformers.tokenization_utils_base import BatchEncoding
from typing import Tuple
from tqdm import tqdm

from utils import create_bias_distribution, check_config, check_attribute_occurence, templates_to_train_samples, \
    templates_to_eval_samples
from embedding import BertHuggingfaceMLM, BertHuggingface
from geometrical_bias import SAME, WEAT, GeneralizedWEAT, DirectBias, RIPA, MAC
from lipstick_bias import BiasGroupTest, NeighborTest, ClusterTest, ClassificationTest
from unmasking_bias import PLLBias, MLMBiasTester, MLMBiasDataset

DEBUG = False


def unmasking_bias(bert: BertHuggingfaceMLM, config: dict, data_test: dict, template_config: dict, target_words: list,
                   group_keys: list, log_dir: str = None) -> dict:
    print("evaluate unmasking bias...")
    mlmBiasTester = MLMBiasTester(bert.model, bert.tokenizer, bert.batch_size)

    # convert test data and remember the template and sample ids (one "sample" refers to a unique template-target
    # combination (but it has several versions for each group of the protected attribute)
    sentences, target_label, attribute_label, group_label, template_ids, sample_ids = [], [], [], [], [], []
    mask_ids = []
    template_id = 0
    sample_id = 0
    for sample in data_test:
        n_groups = len(sample['sentences'])
        sentences += list(sample['sentences'])
        target_label += [sample['target'] for i in range(n_groups)]
        attribute_label += [sample['protected_attr']]
        group_label += list(range(n_groups))
        template_ids += [template_id for i in range(n_groups)]
        sample_ids += [sample_id for i in range(n_groups)]

        if config['eval_strategy'] == 'non_attribute':
            mask_ids += sample['non_attr_token_ids']
        elif config['eval_strategy'] == 'target':
            mask_ids += sample['target_token_ids']
#        elif config['eval_strategy'] == 'attribute':
#            mask_ids += sample['attribute_token_ids']
        else:
            print("eval_strategy", config['eval_strategy'], "is not supported")

        if sample['target'] == target_words[-1]:
            template_id += 1

        sample_id += 1

    # tokenize test sentences
    print("tokenize test sentences...")
    token_ids = mlmBiasTester.tokenizer(sentences, return_tensors='pt', max_length=512, truncation=True,
                                        padding='max_length')
    input_ids = token_ids['input_ids']
    attention_masks = token_ids['attention_mask']

    assert input_ids.size()[0] == len(mask_ids), "mismatch for tokenized samples and mask_ids"

    # we compute biases for each to-be-masked token individually, so create a sample for each token to be masked
    token_ids_single_masks, token_ids_unmasked, single_mask_ids, attention = [], [], [], []
    ref_ids = []  # for each single mask sample, point to the id in the lists above
    for i in range(input_ids.size()[0]):
        for cur_mask_idx in mask_ids[i]:
            masked_token_ids = input_ids[i].clone()
            masked_token_ids[cur_mask_idx] = mlmBiasTester.tokenizer.mask_token_id
            token_ids_single_masks.append(masked_token_ids)
            token_ids_unmasked.append(input_ids[i].clone())
            attention.append(attention_masks[i].clone())
            ref_ids.append(i)
            single_mask_ids.append(cur_mask_idx)

    # convert for batch processing
    encodings = BatchEncoding({'input_ids': token_ids_single_masks, 'label': token_ids_unmasked,
                               'attention_mask': attention, 'mask_ids': single_mask_ids})
    dataset = MLMBiasDataset(encodings)
    loader = torch.utils.data.DataLoader(dataset, batch_size=mlmBiasTester.batch_size, shuffle=True)

    for i in range(len(encodings['mask_ids'])):
        mask_idx = encodings['mask_ids'][i]
        assert encodings['input_ids'][i][mask_idx] == mlmBiasTester.tokenizer.mask_token_id, "to-be-masked token is not masked"
        if config['eval_strategy'] == 'target':
            decoded = mlmBiasTester.tokenizer.decode(encodings['label'][i][mask_idx]).replace('#', '').strip()
            partial_word = False
            for target in target_words:
                if decoded in target or (decoded[-1] == 's' and decoded[:-1] in target):
                    partial_word = True
            assert partial_word or decoded in target_words or decoded[:-1] in target_words, "masked token \""+decoded+"\" is not a target word!"

    # get token probabilities
    token_probs = []
    print("calculate unmasking probabilities for "+str(len(single_mask_ids))+" sentences...")
    loop = tqdm(loader, leave=True)
    for batch_id, sample in enumerate(loop):
        new_token_probs = mlmBiasTester.get_batch_token_probabilities(sample)
        for i, prob in enumerate(new_token_probs):
            if prob > 0.05:
                decoded = mlmBiasTester.tokenizer.decode(sample['label'][i][sample['mask_ids'][i]])
                idx = batch_id*mlmBiasTester.batch_size+i
                target = target_label[ref_ids[idx]]
                if decoded == target:
                    print("got high prob ", prob, "in batch ", batch_id, " for sample ", i, ":", decoded)
                    print(mlmBiasTester.tokenizer.decode(sample['label'][i]).replace(' [PAD]', ''))
        token_probs += new_token_probs

    probs_per_group_target = {}
    log_likelihood_per_group_target = {}
    jsd_per_attr_target = {}
    for target in target_words:
        probs_per_group_target.update({target: {group: [] for group in group_keys}})
        log_likelihood_per_group_target.update({target: {group: [] for group in group_keys}})
        jsd_per_attr_target.update({target: {attr: [] for attr in template_config['protected_attr']}})

    print("compute JSD and log results...")
    for sample_idx in range(sample_id):

        sample_version_ids = [i for i in range(len(single_mask_ids)) if sample_ids[ref_ids[i]] == sample_idx]
        # one list of mask ids per sample version
        all_mask_ids = [mask_ids[ref_ids[i]] for i in range(len(single_mask_ids)) if sample_ids[ref_ids[i]] == sample_idx]
        cur_attr = attribute_label[sample_idx]

        #print(cur_attr)
        #print("all mask ids", all_mask_ids, "for sample: ", data_test[sample_idx]['sentences'])
        #print("sample version ids: ", sample_version_ids)
        #print([sentences[ref_ids[i]] for i in sample_version_ids])

        #print("single mask ids: ", [single_mask_ids[i] for i in sample_version_ids])

        # we except to mask out the unmodified context (or target), so even if we have some offsets in the token ids
        #  due to different numbers of modified tokens, the overall number must be the same!
        for i in range(1, len(all_mask_ids)):
            assert len(all_mask_ids[0]) == len(all_mask_ids[i]), "expected the same number of mask ids for each version of the sample"

        # compute Jensen-Shanon-Divergence per token:
        # token probability normalized over all groups vs. equal distribution
        cur_sample_jsds = []
        for k in range(len(all_mask_ids[0])):
            # this is a sample version id where the current token id is masked:
            ids_for_cur_mask = [idx for i, idx in enumerate(sample_version_ids) if single_mask_ids[idx] == all_mask_ids[i][k]]
            #print("cur mask ids: ", ids_for_cur_mask)
            # this should be one sample per group where the same token was masked
            assert len(ids_for_cur_mask) == len(template_config[attribute_label[sample_idx]][0])

            # normalize probabilities over all groups, then compute JSD to equal distribution
            probs = np.asarray([token_probs[idx] for idx in ids_for_cur_mask])
            probs = probs/np.sum(probs)
            dist_equal = np.ones(probs.shape)/probs.shape[0]
            jsd = scipy.spatial.distance.jensenshannon(probs, dist_equal)
            cur_sample_jsds.append(jsd)
        # mean JSD over all masked tokens for the current sample
        jsd_per_attr_target[target_label[sample_idx]][cur_attr].append(np.mean(cur_sample_jsds))
        # TODO: log all the sample,group,mask-wise results

        # compute overall token likelihood per group over all masked tokens for the current sample
        for group_id, group in enumerate(template_config[cur_attr][0]):
            cur_group_ids = [i for i in sample_version_ids if group_label[ref_ids[i]] == group_id]
            probs = np.asarray([token_probs[idx] for idx in cur_group_ids])
            probs_per_group_target[target_label[sample_idx]][group].append(np.prod(probs))
            #if len(probs) > 1:
            #    print(target_label[ref_ids[sample_version_ids[0]]],
            #          [token_ids_unmasked[idx][single_mask_ids[idx]] for idx in cur_group_ids], probs, np.prod(probs))
            log_likelihood_per_group_target[target_label[sample_idx]][group].append(np.sum(np.log(probs)))
    # TODO: log the dictionaries

    if log_dir is not None:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        log_res = {'token_probs': token_probs, 'input_ids': token_ids_single_masks, 'label': token_ids_unmasked,
                   'attention': attention, 'mask_ids': single_mask_ids,
                   'sentences': [sentences[ref_ids[i]] for i in range(len(token_probs))]}
        with open(log_dir+'/raw_results.pickle', 'wb') as handle:
            pickle.dump(log_res, handle)

        res = {'prob': probs_per_group_target, 'JSD': jsd_per_attr_target,
               'log_likelihood': log_likelihood_per_group_target}
        with open(log_dir+'/all_results.pickle', 'wb') as handle:
            pickle.dump(res, handle)

    # return the mean values per target and attribute/group combination
    unmasking_result = {'JSD': {}, 'prob': {}, 'log_likelihood': {}}
    for target in target_words:
        unmasking_result['JSD'][target] = {attr: np.mean(jsd_per_attr_target[target][attr]) for attr in template_config['protected_attr']}
        unmasking_result['prob'][target] = {group: np.mean(probs_per_group_target[target][group]) for group in group_keys}
        unmasking_result['log_likelihood'][target] = {group: np.mean(log_likelihood_per_group_target[target][group]) for group in group_keys}

    return unmasking_result


def create_defining_embeddings_from_templates(bert, template_config):
    '''
    For each type of attribute, create defining sentences from the templates that include only the respective attribute,
    neutral terms for all other attributes and a masked out target. Defining sentences include identical sentences that
    only differ by the protected group mentioned.
    Returns a dictionary with embeddings of the defining sentences (list of lists) by attribute keys.
    '''
    templates = template_config['templates_test']
    attributes = template_config['protected_attr']

    emb_dict = {}
    for attr in attributes:
        emb_dict.update({attr: []})

    for temp in templates:
        for attr in attributes:
            if attr not in temp:
                continue
            sent = temp

            # replace all other attributes with the neutral term
            for attr2 in attributes:
                if attr2 == attr:
                    continue
                for i in range(len(template_config[attr2]) - 1, -1, -1):
                    cur_attr = attr2 + str(i)
                    sent = sent.replace(cur_attr, template_config[attr2 + '_neutral'][i])

            # replace target key by mask
            sent = sent.replace(template_config['target'], '[MASK]')

            # for each group, create a sentence where the current attribute is replaced by this group
            group_versions = []
            for k, group in enumerate(template_config[attr][0]):
                sent2 = sent
                for i in range(len(template_config[attr]) - 1, -1, -1):
                    cur_attr = attr + str(i)
                    sent2 = sent2.replace(cur_attr, template_config[attr][i][k])
                group_versions.append(sent2)
            emb = bert.embed(group_versions)
            emb_dict[attr].append(emb)

    return emb_dict


def report_bias_scores(bert: BertHuggingfaceMLM, defining_emb: dict, data_test: list, target_words: list,
                       groups_per_attr: dict, target_stat_df: pd.DataFrame):

    bias_score = [SAME(), MAC(), DirectBias(), RIPA(), WEAT(), GeneralizedWEAT(), WEAT(), GeneralizedWEAT()]#,
#                  ClusterTest(), ClassificationTest(), NeighborTest(k=100), ClusterTest(), ClassificationTest(),
#                  NeighborTest(k=100)]
    score_names = ["SAME", "MAC", "DirectBias", "RIPA", "WEAT", "GWEAT", "WEAT_i", "GWEAT_i"]#, "cluster", "classification", "neighbor",
#                   "cluster_i", "classification_i", "neighbor_i"]
    groups = list(target_stat_df.index)

    # lookup for the majority group of each target by attribute
    group_label_by_attr = {}  # labels with some noise (assuming biases in the data do not correspond exactly to biases in society/ assumptions of the user)
    group_label_by_attr_i = {}  # ideal labels (exact knowledge of biases in the data)
    for attr in groups_per_attr.keys():
        cur_groups = groups_per_attr[attr]
        attr_probs = target_stat_df.loc[cur_groups]

        mu, sigma = 0, 0.01
        noise = np.random.normal(mu, sigma, attr_probs.shape)
        attr_probs_noise = attr_probs.to_numpy()+noise
        group_label = np.argmax(attr_probs_noise, axis=0)
        group_label_i = np.argmax(attr_probs.to_numpy(), axis=0)
        print("ideal vs. noisy group label alignment:")
        print(np.sum([1 for i in range(len(group_label)) if group_label[i] == group_label_i[i]])/len(group_label))
        group_label_by_attr.update({attr: {}})
        group_label_by_attr_i.update({attr: {}})
        for i, target in enumerate(target_words):
            group_label_by_attr[attr].update({target: group_label[i]})
            group_label_by_attr_i[attr].update({target: group_label_i[i]})

    biases_by_target_attr = {}
    biases_by_score_attr = {}
    pair_biases_by_target_group = {}
    for group in groups:
        pair_biases_by_target_group.update({group: {}})
        pair_biases_by_target_group[group].update({target: [] for target in target_words})

    for score in score_names:
        biases_by_score_attr.update({score: {attr: {} for attr in defining_emb.keys()}})
        biases_by_target_attr.update({score: {}})
        biases_by_target_attr[score].update({attr: {target: [] for target in target_words} for attr in defining_emb.keys()})

    for cur_attr_key, embeddings in defining_emb.items():
        sel_texts = []
        sel_targets = []
        sel_mask_tokens = []
        y = []  # group assignment with noise
        y_i = []  # ideal group assignment
        for sample in data_test:
            sentence_versions = sample['sentences']
            attr_token_ids = sample['attribute_token_ids']
            if sample['protected_attr'] == cur_attr_key:
                # different attributes may have different numbers of tokens, so we may need to add multiple masked
                # versions of the current sentence
                token_nums_added = []
                for i in range(len(sentence_versions)):
                    if len(attr_token_ids[i]) not in token_nums_added:
                        token_nums_added.append(len(attr_token_ids[i]))
                        sel_texts.append(sentence_versions[i])
                        sel_targets.append(sample['target'])
                        sel_mask_tokens.append(attr_token_ids[i])
                        y.append(group_label_by_attr[sample['protected_attr']][sample['target']])
                        y_i.append(group_label_by_attr_i[sample['protected_attr']][sample['target']])

        assert len(sel_texts) > 0, "there are no test sentences for attribute "+cur_attr_key

        # mask out all attribute tokens, then embed
        token_ids = bert.tokenizer(sel_texts)['input_ids']
        for i in range(len(sel_mask_tokens)):
            for j in sel_mask_tokens[i]:
                token_ids[i][j] = bert.tokenizer.mask_token_id
        masked_sent = [bert.tokenizer.decode(ids[1:-1]) for ids in token_ids]
        sel_emb = bert.embed(masked_sent)

        emb_lists = []
        for c in range(max(y)+1):
            c_emb = [sel_emb[i] for i in range(len(sel_texts)) if y[i] == c]
            emb_lists.append(c_emb)
            print("emb list for group", groups_per_attr[cur_attr_key][c], "has len", len(c_emb))

        for idx, score in enumerate(bias_score):
            print("compute bias score: "+score_names[idx])
            binary_score = (("WEAT" in score_names[idx] and "GWEAT" not in score_names[idx]) or "cluster" in score_names[idx])
            if binary_score and len(embeddings) > 2:
                continue

            cur_y = y
            score_name = score_names[idx]
            score_name_short = score_name
            if "_i" in score_name:
                score_name_short = score_name[:-2]
                cur_y = y_i

            if score_name_short not in ['cluster', 'classification', 'neighbor']:
                score.define_bias_space(embeddings)

            # individual bias scores (SAME, WEAT, MAC, DirectBias, RIPA)
            if score_name_short in ['SAME', 'WEAT', 'MAC', 'DirectBias', 'RIPA']:
                for i, target in enumerate(sel_targets):
                    if score_name == "SAME":
                        pair_biases = score.individual_bias_per_pair(sel_emb[i])
                        pair_biases_by_target_group[groups_per_attr[cur_attr_key][0]][target].append(0)
                        for j in range(1, len(groups_per_attr[cur_attr_key])):
                            pair_biases_by_target_group[groups_per_attr[cur_attr_key][j]][target].append(pair_biases[j-1])
                    if score_name == 'SAME' and len(embeddings) == 2:
                        biases_by_target_attr[score_name][cur_attr_key][target].append(score.signed_individual_bias(sel_emb[i]))
                    else:
                        biases_by_target_attr[score_name][cur_attr_key][target].append(score.individual_bias(sel_emb[i]))

            # overall bias scores (cosine scores and lipstick tests
            if score_name_short in ["WEAT", "GWEAT"]:
                biases_by_score_attr[score_name][cur_attr_key] = score.group_bias(emb_lists)
            elif score_name_short == "cluster":
                biases_by_score_attr[score_name][cur_attr_key] = score.cluster_test_with_labels(sel_emb, cur_y)
            elif score_name_short == "classification":
                biases_by_score_attr[score_name][cur_attr_key] = np.mean(score.classification_test_with_labels(sel_emb, cur_y))
            elif score_name_short == "neighbor":
                biases_by_score_attr[score_name][cur_attr_key] = np.mean(score.bias_by_neighbor(emb_lists))
            else:
                # mean individual bias
                biases_by_score_attr[score_name][cur_attr_key] = score.mean_individual_bias(sel_emb)

    # return mean biases as dataframes
    dataframes = {}
    for score in score_names:
        mean_bias_by_target = {}
        for attr_key, v in biases_by_target_attr[score].items():
            mean_bias_by_target.update({attr_key: {}})
            for target in target_words:
                mean_bias_by_target[attr_key].update({target: np.mean(v[target])})
        df = pd.DataFrame(data=mean_bias_by_target)
        dataframes.update({score: df})

    df_overall = pd.DataFrame(data=biases_by_score_attr)
    print(df_overall)

    mean_pair_bias_by_target = {}
    for attr_key, v in pair_biases_by_target_group.items():
        mean_pair_bias_by_target.update({attr_key: {}})
        for target in target_words:
            mean_pair_bias_by_target[attr_key].update({target: np.mean(v[target])})

    df2 = pd.DataFrame(data=mean_pair_bias_by_target)

    return dataframes, score_names, df2, df_overall


def data_model_bias_corr(stat_path: str, unmasking_results: dict, template_config: dict) -> Tuple[float, float]:
    df = pd.read_csv(stat_path)
    all_data_bias = []
    all_task_bias = []

    # compute JSD on target-group probs in the training data
    data_jsd = {}
    attributes = template_config['protected_attr']
    targets = template_config[template_config['target']]
    groups = [group for attr in attributes for group in template_config[attr][0]]
    for target in targets:
        data_jsd[target] = {}

        for attr in attributes:
            group_ids = [groups.index(group) for group in template_config[attr][0]]
            probs = np.array(df.loc[group_ids[0]:group_ids[-1], target])
            equal_dist = np.ones(probs.shape)/probs.shape
            data_jsd[target][attr] = scipy.spatial.distance.jensenshannon(probs, equal_dist)
    df_data = pd.DataFrame(data_jsd)

    df_task = pd.DataFrame(unmasking_results['JSD'])

    print("data:")
    print(df_data)
    print("task:")
    print(df_task)
    assert df_data.shape == df_task.shape, "expected the same shape of results for data and task biases!"

    for i in range(df_data.shape[1]):
        if df_data.columns[i] not in targets:
            continue
        data_bias = list(df_data.loc[:, df_data.columns[i]])
        pretrain_bias = list(df_task.loc[:, df_data.columns[i]])
        all_data_bias += data_bias
        all_task_bias += pretrain_bias

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(all_data_bias,
                                                                         all_task_bias)
    print("data - eval r2: ", r_value, "(", p_value, ")")

    return r_value, p_value


def run(config, min_iter=0, max_iter=-1):

    print("load templates and protected attributes...")
    with open(config['template_file'], 'r') as f:
        template_config = yaml.safe_load(f)

    target_domain = template_config['target']
    target_words = template_config[target_domain]
    protected_attributes = template_config['protected_attr']

    protected_groups = {}
    group_attr = []
    for attr in protected_attributes:
        protected_groups.update({attr: template_config[attr][0]})
        for i in range(len(template_config[attr])):
            group_attr += template_config[attr][i]

    check_attribute_occurence(template_config)

    print("create the datasets for all experiment iterations...")
    if not os.path.isdir(config['results_dir']):
        os.makedirs(config['results_dir'])
    log_config = config['results_dir']+'/config.yaml'
    print(config)

    with open(log_config, 'w') as file:
        yaml.dump(config, file)

    print("minP choices: ", config['minP'])
    print("maxP choices: ", config['maxP'])
    print("iterations: ", config['iterations'])
    iter_id = -1
    iter_lookup = {}
    for minP in config['minP']:
        for maxP in config['maxP']:

            probs_by_attr = {}
            for attr, groups in protected_groups.items():
                n = len(groups)
                res = create_bias_distribution(n, target_words, minP=minP / n, maxP=maxP / n)
                df = pd.DataFrame(data=res)
                probs_by_attr.update({attr: df})

            for it in range(config['iterations']):
                iter_id += 1
                if iter_id < min_iter or (iter_id > max_iter and not max_iter == -1):
                    continue
                print("handling model ", iter_id, "with params:")
                print("minP:", minP, "maxP: ", maxP, "iteration: ", it)
                iter_results = config['results_dir'] + '/' + str(iter_id)
                if not os.path.exists(iter_results):
                    os.makedirs(iter_results)
                iter_lookup.update({iter_id: (minP, maxP, it)})
                model_path = iter_results+'/model'
                data_path = iter_results+'/data.pickle'
                eval_detailed_results_path = iter_results+'/eval_details/'
                stat_path = iter_results+'/train_data_stats.csv'
                model_bias_path = iter_results + "/task_res.csv"
                iter_config = {'minP': minP, 'maxP': maxP, 'iteration': it, 'base_config': log_config,
                               'model': model_path, 'data': data_path, 'stat': stat_path}
                config_file = iter_results+'/config.yaml'
                with open(config_file, 'w') as file:
                    yaml.dump(iter_config, file)

                groups = [group for attr in protected_attributes for group in template_config[attr][0]]

                data_exists = os.path.isfile(data_path)
                checkpoint_exists = data_exists and os.path.isdir(model_path)

                # load pretrained model
                if 'MLM' in config['objective']:
                    bert = BertHuggingfaceMLM(model_name=config['pretrained_model'], batch_size=config['batch_size'])
                else:
                    bert = BertHuggingface(model_name=config['pretrained_model'], batch_size=config['batch_size'],
                                           num_labels=2)
                # end loading model

                # creating or loading the dataset
                if not data_exists:
                    print("create dataset from templates with minP and maxP parameters and save it...")

                    data_train = templates_to_train_samples(bert.tokenizer, template_config, probs_by_attr,
                                                            target_words, config)
                    data_test = templates_to_eval_samples(bert.tokenizer, template_config, target_words)
                    if DEBUG:
                        data_test = data_test[:10*len(target_words)]
                    data_save = {'train': data_train, 'test': data_test, 'epochs': config['epochs']}

                    with open(data_path, "wb") as handler:
                        pickle.dump(data_save, handler)

                    print("log co-occurence of target words and protected groups...")
                    target_group_occ = {}
                    for target in target_words:
                        target_group_occ[target] = {group: 0 for group in groups}

                    for sample in data_train:
                        for attr in protected_attributes:
                            if sample[attr] > -1:  # group id ( > -1 if attribute exists)
                                target_group_occ[sample['target']][protected_groups[attr][sample[attr]]] += 1

                    df = pd.DataFrame(data=target_group_occ)

                    # normalize per group ( -> p(target | group))
                    for group in groups:
                        # overall occurence of this group
                        sel = df.loc[group, :]
                        sel_sum = np.sum(sel)
                        # normalize
                        df.loc[group, :] /= sel_sum

                    print(df)
                    df.to_csv(stat_path, index_label='groups')

                else:
                    print("load training data from "+data_path)
                    with open(data_path, "rb") as handler:
                        data_save = pickle.load(handler)
                # end of dataset creation/ loading

                # model training and validation
                training_iterations = config['max_retries']
                saved_r = -1
                training_done = False
                data_test = data_save['test']

                if checkpoint_exists:
                    saved_r = data_save['baseline_r2']
                    if 'iter_left' in data_save.keys():
                        training_iterations = data_save['iter_left']
                        print("need to resume training with", training_iterations, "iterations left")

                    training_done = training_iterations == 0 or saved_r >= config['target_r_value']

                if training_done:
                    print("load model from checkpoint...")
                    bert.load(model_path)
                else:
                    data_train = data_save['train']
                    X_train = [sample['masked_sentence'] for sample in data_train]
                    y_train = [sample['sentence'] for sample in data_train]

                    if DEBUG:
                        X_train = X_train[:500]
                        y_train = y_train[:500]
                    print("retrain BERT with ", len(X_train), " training samples for ", config['epochs'], " epochs")

                    r_value = saved_r
                    last_r_value = saved_r
                    it = 0

                    while r_value < config['target_r_value'] and it < training_iterations:
                        print("iteration", it, "(of max", training_iterations, "iterations)")

                        if 'MLM' in config['objective']:
                            bert = BertHuggingfaceMLM(model_name=config['pretrained_model'],
                                                      batch_size=config['batch_size'])

                            losses = []
                            detailed_results_dir = None
                            unmasking_results = None
                            for ep in range(config['epochs']):
                                print("at epoch "+str(ep))
                                epoch_log_dir = iter_results + '/epoch'+str(ep)
                                if not os.path.isdir(epoch_log_dir):
                                    os.makedirs(epoch_log_dir)
                                losses += bert.retrain(X_train, y_train, epochs=1)
                                if ep == config['epochs']-1:
                                    detailed_results_dir = eval_detailed_results_path

                                attributes = template_config['protected_attr']
                                embeddings = {'targets': bert.embed(target_words),
                                              'attributes': {attr: [bert.embed(words) for words in template_config[attr]] for attr in attributes}}
                                with open(epoch_log_dir+'/emb.pickle', 'wb') as handler:
                                    pickle.dump(embeddings, handler)

                                unmasking_results = unmasking_bias(bert, config, data_test, template_config,
                                                                   target_words, groups,
                                                                   log_dir=detailed_results_dir)
                                r_value, p_value = data_model_bias_corr(stat_path, unmasking_results, template_config)
                                print("after epoch "+str(ep)+" got R value: "+str(r_value)+"("+str(p_value)+")")
                                with open(epoch_log_dir+'/results.pickle', 'wb') as handler:
                                    pickle.dump(unmasking_results, handler)
                        else:
                            print("objective not supported, please select 'MLM' or 'MLM_lazy'")
                            exit(1)

                        df = pd.DataFrame(unmasking_results['prob'])
                        print(df)
                        df.to_csv(model_bias_path, index_label='groups')
                        # TODO: correlation can only be determined over data JSD/ mean prediction JSD, because
                        #  token probs vary too much between single-token words and multi-token words
                        #  (so we cant trust mean probs)
                        r_value, p_value = data_model_bias_corr(stat_path, unmasking_results, template_config)

                        it += 1
                        if r_value > last_r_value:
                            print("save model with r_value of ", r_value)
                            bert.save(model_path)

                            with open(iter_results + '/eval_results.pickle', 'wb') as handler:
                                pickle.dump(unmasking_results, handler)

                            data_save['loss'] = losses
                            data_save['iter_left'] = training_iterations-it
                            data_save['baseline_r2'] = r_value
                            print("iterations left: ", data_save['iter_left'])

                            with open(data_path, "wb") as handler:
                                pickle.dump(data_save, handler)

                        last_r_value = r_value
                    print("done with training, final r_value: ", r_value)
                    data_save['iter_left'] = 0

                    with open(data_path, "wb") as handler:
                        pickle.dump(data_save, handler)
                # end model training and validation

                # evaluate model for biases
                print("evaluate biases on training task...")
                assert len(data_test) > 0, "got no sentences for bias test evaluation"

                # test cosine scores on the masked sentences
                def_emb = create_defining_embeddings_from_templates(bert, template_config)
                for k, v in def_emb.items():
                    v2 = list(zip(*v))
                    def_emb[k] = []
                    for tup in v2:
                        def_emb[k].append(np.asarray(tup))

                df_train = pd.read_csv(stat_path, index_col='groups')
                res, scores, df_pair_bias, df_overall = report_bias_scores(bert, def_emb, data_test, target_words,
                                                                           protected_groups, df_train)
                # end evaluation

                # save results
                data_save['overall_biases'] = df_overall
                for score in scores:
                    data_save[score+'_bias'] = res[score]
                data_save['same_pair_bias'] = df_pair_bias
                print("data save keys:")
                print(data_save.keys())

                with open(data_path, "wb") as handler:
                    print("save data")
                    pickle.dump(data_save, handler)


def main(argv):
    config_path = ''
    min_iter = 0
    max_iter = -1
    try:
        opts, args = getopt.getopt(argv, "hc:", ["config=", "min=", "max="])
    except getopt.GetoptError:
        print('multi_attr_bias_test.py -c <config>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('multi_attr_bias_test.py -c <config>')
            sys.exit()
        elif opt in ("-c", "--config"):
            config_path = arg
        elif opt == "--min":
            min_iter = int(arg)
        elif opt == "--max":
            max_iter = int(arg)

    print('config is ' + config_path)

    with open(config_path, 'rb') as f:
        config = yaml.safe_load(f)
        check_config(config)
    print(config)

    run(config, min_iter, max_iter)


if __name__ == "__main__":
    main(sys.argv[1:])
