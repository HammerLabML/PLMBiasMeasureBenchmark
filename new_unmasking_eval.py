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

from utils import create_bias_distribution, check_config, check_attribute_occurence, templates_to_train_samples, \
    templates_to_eval_samples
from embedding import BertHuggingfaceMLM, BertHuggingface
from geometrical_bias import SAME, WEAT, GeneralizedWEAT, DirectBias, RIPA, MAC
from lipstick_bias import BiasGroupTest, NeighborTest, ClusterTest, ClassificationTest
from unmasking_bias import PLLBias, MLMBiasTester, MLMBiasDataset
DEBUG = True


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
        elif config['eval_strategy'] == 'attribute':
            mask_ids += sample['attribute_token_ids']
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
        for j in range(len(mask_ids[i])):
            masked_token_ids = input_ids[i].clone()
            masked_token_ids[j] = mlmBiasTester.tokenizer.mask_token_id
            token_ids_single_masks.append(masked_token_ids)
            token_ids_unmasked.append(input_ids[i].clone())
            attention.append(attention_masks[i].clone())
            ref_ids.append(i)
            single_mask_ids.append(j)

    # convert for batch processing
    encodings = BatchEncoding({'input_ids': token_ids_single_masks, 'label': token_ids_unmasked,
                               'attention_mask': attention, 'mask_ids': single_mask_ids})
    dataset = MLMBiasDataset(encodings)
    loader = torch.utils.data.DataLoader(dataset, batch_size=mlmBiasTester.batch_size, shuffle=True)

    # get token probabilities
    token_probs = []
    print("calculate unmasking probabilities...")
    for batch_id, sample in enumerate(loader):
        token_probs += mlmBiasTester.get_batch_token_probabilities(sample)

    probs_per_group_target = {}
    log_likelihood_per_group_target = {}
    jsd_per_attr_target = {}
    for target in target_words:
        probs_per_group_target.update({target: {group: [] for group in group_keys}})
        log_likelihood_per_group_target.update({target: {group: [] for group in group_keys}})
        jsd_per_attr_target.update({target: {attr: [] for attr in template_config['protected_attr']}})

    print("compute JSD and log results...")
    for sample_idx in range(sample_id):
        sample_version_ids = [i for i in range(len(token_probs)) if ref_ids[i] == sample_idx]
        all_mask_ids = mask_ids[sample_idx]
        cur_attr = attribute_label[sample_idx]

        # compute Jensen-Shanon-Divergence per token:
        # token probability normalized over all groups vs. equal distribution
        cur_sample_jsds = []
        for k in all_mask_ids:
            cur_mask_ids = [i for i in sample_version_ids if single_mask_ids[i] == k]
            # this should be one sample per group where the same token was masked
            print(cur_mask_ids)
            print([sentences[ref_ids[i]] for i in sample_version_ids])
            assert len(cur_mask_ids) == len(template_config[attribute_label[sample_idx]][0])

            # normalize probabilities over all groups, then compute JSD to equal distribution
            probs = np.asarray([token_probs[idx] for idx in cur_mask_ids])
            probs = probs/np.sum(probs)
            dist_equal = np.ones(probs.shape)/probs.shape[0]
            jsd = scipy.spatial.distance.jensenshannon(probs, dist_equal)
            cur_sample_jsds.append(jsd)
        # mean JSD over all masked tokens for the current sample
        jsd_per_attr_target[target_label[sample_idx]][cur_attr] = np.mean(cur_sample_jsds)
        # TODO: log all the sample,group,mask-wise results

        # compute overall token likelihood per group over all masked tokens for the current sample
        for group_id, group in enumerate(template_config[cur_attr][0]):
            cur_group_ids = [i for i in sample_version_ids if group_label[ref_ids[i]] == group_id]
            probs = np.asarray([token_probs[idx] for idx in cur_group_ids])
            probs_per_group_target[target_label[sample_idx]][group] = np.prod(probs)
            log_likelihood_per_group_target[target_label[sample_idx]][group] = np.sum(np.log(probs))
    # TODO: log the dictionaries

    # return the mean values per target and attribute/group combination
    unmasking_result = {'JSD': {}, 'prob': {}, 'log_likelihood': {}}
    for target in target_words:
        unmasking_result['JSD'][target] = {attr: np.mean(jsd_per_attr_target[target][attr]) for attr in template_config['protected_attr']}
        unmasking_result['prob'][target] = {group: np.mean(probs_per_group_target[target][group]) for group in group_keys}
        unmasking_result['log_likelihood'][target] = {group: np.mean(probs_per_group_target[target][group]) for group in group_keys}

    return unmasking_result


def data_model_bias_corr(stat_path: str, unmasking_result: dict) -> Tuple[float, float]:
    df = pd.read_csv(stat_path)
    all_data_bias = []
    all_task_bias = []

    df_task = pd.DataFrame(unmasking_result['prob'])

    assert df.shape == df_task.shape, "expected the same shape of results for data and task biases!"

    for i in range(df.shape[1]):
        if df.columns[i] == 'Unnamed: 0':
            continue
        data_bias = list(df.loc[:, df.columns[i]])
        pretrain_bias = list(df_task.loc[:, df.columns[i]])
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
                data_path = iter_results+'/train_data.pickle'
                eval_detailed_results_path = iter_results+'/eval_details/'
                stat_path = iter_results+'/train_data_stats.csv'
                model_bias_path = iter_results + "/task_res.csv"
                iter_config = {'minP': minP, 'maxP': maxP, 'iteration': it, 'base_config': log_config,
                               'model': model_path, 'data': data_path, 'stat': stat_path}
                config_file = iter_results+'/config.yaml'
                with open(config_file, 'w') as file:
                    yaml.dump(iter_config, file)

                target_group_occ = {}

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
                    data_save = {'train': data_train, 'test': data_test, 'epochs': config['epochs']}

                    with open(data_path, "wb") as handler:
                        pickle.dump(data_save, handler)

                    print("log co-occurence of target words and protected groups...")
                    groups = [group for attr in protected_attributes for group in template_config[attr][0]]
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
                    df.to_csv(stat_path)

                else:
                    print("load training data from "+data_path)
                    with open(data_path, "rb") as handler:
                        data_save = pickle.load(handler)

                    df = pd.read_csv(stat_path)
                # end of dataset creation/ loading

                # model training and validation
                training_iterations = config['max_retries']
                saved_r = -1
                training_done = False
                data_test = data_save['test']

                if checkpoint_exists:
                    target_label = data_save['target_label']
                    attr_label = data_save['attr_label']
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
                                                                   target_words, list(target_group_occ.keys()),
                                                                   log_dir=detailed_results_dir)
                                with open(epoch_log_dir+'/results.pickle', 'wb') as handler:
                                    pickle.dump(unmasking_results, handler)
                        else:
                            print("objective not supported, please select 'MLM' or 'MLM_lazy'")
                            exit(1)

                        r_value, p_value = data_model_bias_corr(stat_path, unmasking_results)

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
"""
                # evaluate model for biases
                print("evaluate biases on training task...")
                assert len(test_data) > 0, "got no sentences for bias test evaluation"

                # test cosine scores on the masked sentences
                def_emb = create_defining_embeddings_from_templates(bert, tmp)
                for k, v in def_emb.items():
                    v2 = list(zip(*v))
                    def_emb[k] = []
                    for tup in v2:
                        def_emb[k].append(np.asarray(tup))
                dataframes, score_names, df_pair_bias, df_overall = report_bias_scores(bert, def_emb, test_data, attr_label, target_label,
                                                                                       target_words, target_group_occ.keys(), protected_groups, df)
                # end evaluation

                # save results
                data_save['overall_biases'] = df_overall
                for score in score_names:
                    data_save[score+'_bias'] = dataframes[score]
                data_save['same_pair_bias'] = df_pair_bias
                print("data save keys:")
                print(data_save.keys())

                with open(data_path, "wb") as handler:
                    print("save data")
                    pickle.dump(data_save, handler)
"""


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
