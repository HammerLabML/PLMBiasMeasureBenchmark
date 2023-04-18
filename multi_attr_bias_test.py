import numpy as np
import os
import pandas as pd
import pickle
import random
import scipy
import yaml
import getopt
import sys

import torch
from transformers import pipeline
from utils import create_bias_distribution, check_config
from embedding import BertHuggingfaceMLM, BertHuggingface
from geometrical_bias import SAME, WEAT, GeneralizedWEAT, DirectBias, RIPA, MAC
from lipstick_bias import BiasGroupTest, NeighborTest, ClusterTest, ClassificationTest

DEBUG = False


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
                for i in range(len(template_config[protected_attr])-1, -1, -1):
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


# returns the unmasking bias for different groups regarding 1 protected attribute, which is masked out in the sentence
def unmasking_bias(unmasker, masked_sent, group_tokens):
    result = unmasker(masked_sent, targets=group_tokens, top_k=len(group_tokens))

    prob = 0
    for res in result:
        prob += res['score']

    probs = []
    for token in group_tokens:
        for res in result:
            if res['token_str'] == token:
                probs.append(res['score'] / prob)
    return probs


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


def report_bias_scores(bert, defining_emb, masked_texts, attr_label, target_label, target_words, groups, protected_groups, target_stat_df):
    bias_score = [SAME(), MAC(), DirectBias(), RIPA(), WEAT(), GeneralizedWEAT(), ClusterTest(), ClassificationTest(), NeighborTest(k=100), WEAT(), GeneralizedWEAT(), ClusterTest(), ClassificationTest(), NeighborTest(k=100)]
    score_names = ["SAME", "MAC", "DirectBias", "RIPA", "WEAT", "GWEAT", "cluster", "classification", "neighbor", "WEAT_i", "GWEAT_i", "cluster_i", "classification_i", "neighbor_i"]

    # lookup for the majority group of each target by attribute
    group_label_by_attr = {}  # labels with some noise (assuming biases in the data do not correspond exactly to biases in society/ assumptions of the user)
    group_label_by_attr_i = {}  # ideal labels (exact knowledge of biases in the data)
    for attr in protected_groups.keys():
        cur_groups = protected_groups[attr]
        attr_probs = target_stat_df.loc[:, cur_groups]

        mu, sigma = 0, 0.3
        noise = np.random.normal(mu, sigma, attr_probs.shape)
        attr_probs_noise = attr_probs.to_numpy()+noise
        group_label = np.argmax(attr_probs_noise, axis=1)
        group_label_i = np.argmax(attr_probs.to_numpy(), axis=1)
        print("ideal vs. noisy group labels:")
        print(group_label_i)
        print(group_label)
        group_label_by_attr.update({attr: {}})
        group_label_by_attr_i.update({attr: {}})
        for i, target in enumerate(target_words):
            group_label_by_attr[attr].update({target: group_label[i]})
            group_label_by_attr_i[attr].update({target: group_label[i]})

    biases_by_target_attr = {}
    biases_by_score_attr = {}
    pair_biases_by_target_group = {}
    for group in groups:
        pair_biases_by_target_group.update({group: {}})
        for target in target_words:
            pair_biases_by_target_group[group].update({target: []})

    for score in score_names:
        biases_by_score_attr.update({score: {}})
        biases_by_target_attr.update({score: {}})
        for attr in defining_emb.keys():
            biases_by_score_attr[score].update({attr: {}})
            biases_by_target_attr[score].update({attr: {}})
            for target in target_words:
                biases_by_target_attr[score][attr].update({target: []})

    for attr_key, embeddings in defining_emb.items():

        sel_texts = []
        sel_targets = []
        y = []  # group assignment with noise
        y_i = []  # ideal group assignment
        for i, text in enumerate(masked_texts):
            if attr_label[i] == attr_key:
                sel_texts.append(text)
                sel_targets.append(target_label[i])
                y.append(group_label_by_attr[attr_label[i]][target_label[i]])
                y_i.append(group_label_by_attr_i[attr_label[i]][target_label[i]])

        if len(sel_texts) > 0:
            sel_emb = bert.embed(sel_texts)

        emb_lists = []
        print(protected_groups[attr_key])
        for c in range(max(y)+1):
            c_emb = [sel_emb[i] for i in range(len(sel_texts)) if y[i] == c]
            emb_lists.append(c_emb)
            print("emb list for group", protected_groups[attr_key][c], "has len", len(c_emb))

        for idx, score in enumerate(bias_score):
            if (("WEAT" in score_names[idx] and not "GWEAT" in score_names[idx]) or "cluster" in score_names[idx]) and len(embeddings) > 2:
                continue

            cur_y = y
            score_name = score_names[idx]
            score_name_short = score_name
            print(score_name)
            if "_i" in score_name:
                score_name_short = score_name[:-2]

            if score_name_short not in ['cluster', 'classification', 'neighbor']:
                score.define_bias_space(embeddings)

            # individual bias scores (SAME, WEAT, MAC, DirectBias, RIPA)
            if score_name_short in ['SAME', 'WEAT', 'MAC', 'DirectBias', 'RIPA']:
                for i, target in enumerate(sel_targets):
                    if score_name == "SAME":
                        pair_biases = score.individual_bias_per_pair(sel_emb[i])
                        pair_biases_by_target_group[protected_groups[attr_key][0]][target].append(0)
                        for j in range(1, len(protected_groups[attr_key])):
                            pair_biases_by_target_group[protected_groups[attr_key][j]][target].append(pair_biases[j-1])
                    if score_name == 'SAME' and len(embeddings) == 2:
                        biases_by_target_attr[score_name][attr_key][target].append(score.signed_individual_bias(sel_emb[i]))
                    else:
                        biases_by_target_attr[score_name][attr_key][target].append(score.individual_bias(sel_emb[i]))

            # overall bias scores (cosine scores and lipstick tests
            if score_name_short in ["WEAT", "GWEAT"]:
                biases_by_score_attr[score_name][attr_key] = score.group_bias(emb_lists)
            elif score_name_short == "cluster":
                biases_by_score_attr[score_name][attr_key] = score.cluster_test_with_labels(sel_emb, cur_y)
            elif score_name_short == "classification":
                biases_by_score_attr[score_name][attr_key] = np.mean(score.classification_test_with_labels(sel_emb, cur_y))
            elif score_name_short == "neighbor":
                biases_by_score_attr[score_name][attr_key] = np.mean(score.bias_by_neighbor(emb_lists))
            else:
                # mean individual bias
                biases_by_score_attr[score_name][attr_key] = score.mean_individual_bias(sel_emb)

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


def unmasking_bias_multi_attr(bert, template_config, target_words, groups):
    templates = template_config['templates_test']
    attributes = template_config['protected_attr']
    probabilities = []
    masked_sentences = []
    attr_label = []
    target_label = []

    group_token_by_attr = {}
    attr_results = {}
    for attr in attributes:
        group_token_by_attr.update({attr: []})
        attr_results.update({attr: {}})
        for i in range(len(template_config[attr])):
            group_token_by_attr[attr].append(template_config[attr][i])

    probs_by_target_group = {}
    for group in groups:
        probs_by_target_group.update({group: {}})
        for target in target_words:
            probs_by_target_group[group].update({target: []})

    if torch.cuda.is_available():
        unmasker = pipeline('fill-mask', model=bert.model, tokenizer=bert.tokenizer, device=0)
    else:
        unmasker = pipeline('fill-mask', model=bert.model, tokenizer=bert.tokenizer, device=-1)

    for temp in templates:
        for attr in attributes:
            # count back in case there are more than 10 versions of this attribute (e.g. GENDER10 contains GENDER1)
            sent = temp

            # replace all other attributes with the neutral term
            for attr2 in attributes:
                if attr2 == attr:
                    continue
                for i in range(len(template_config[attr2]) - 1, -1, -1):
                    cur_attr = attr2 + str(i)
                    sent = sent.replace(cur_attr, template_config[attr2 + '_neutral'][i])

            # now insert the mask for the targeted attribute
            for i in range(len(template_config[attr]) - 1, -1, -1):
                cur_attr = attr + str(i)
                if cur_attr not in sent:
                    continue
                sent2 = sent

                sent2 = sent2.replace(cur_attr, '[MASK]')
                # in case there are multiple words defining this attribute, replace others with the neutral term
                for j in range(len(template_config[attr]) - 1, -1, -1):
                    if not j == i:
                        sent2 = sent2.replace(attr + str(j), template_config[attr + '_neutral'][j])

                # replace target and obtain unmasking probabilities for each group per target
                for target in target_words:
                    masked_sent = sent2.replace(template_config['target'], target)

                    if not masked_sent.count('[MASK]') == 1:
                        print("zero or mulitple masks in sentence!")
                        print(masked_sent)
                        print(sent)
                        print(cur_attr)
                    probs = unmasking_bias(unmasker, masked_sent, group_token_by_attr[attr][i])
                    masked_sentences.append(masked_sent)
                    attr_label.append(attr)
                    target_label.append(target)
                    probabilities.append(probs)

                    for k, group in enumerate(group_token_by_attr[attr][0]):
                        if not group in probs_by_target_group.keys():
                            print("err group not in keys: ", group, probs_by_target_group.keys())
                        if not target in probs_by_target_group[group].keys():
                            print("err target not in keys: ", target, probs_by_target_group[group].keys())
                        if k >= len(probs):
                            print("k exceeds probs: ", k, probs)
                        probs_by_target_group[group][target].append(probs[k])

                # if there are other versions of this attribute, this will be replaced with the neutral term anyways
                sent = sent.replace(cur_attr, template_config[attr + '_neutral'][i])

    mean_prob_by_target_group = {}
    for group in groups:
        mean_prob_by_target_group.update({group: {}})
        for target in target_words:
            mean_prob_by_target_group[group].update({target: np.mean(probs_by_target_group[group][target])})

    df = pd.DataFrame(data=mean_prob_by_target_group)

    return df, probabilities, masked_sentences, attr_label, target_label


def generate_sentences_for_mlm(template_config, target_words):
    templates = template_config['templates_test']
    attributes = template_config['protected_attr']

    test_data = []
    attr_label = []
    target_label = []
    for temp in templates:
        for attr in attributes:
            if attr not in sent:
                continue
            sent = temp

            # replace all other attributes with the neutral term
            for attr2 in attributes:
                if attr2 == attr:
                    continue
                # count back in case there are more than 10 versions of this attribute (e.g. GENDER10 contains GENDER1)
                for i in range(len(template_config[attr2]) - 1, -1, -1):
                    cur_attr = attr2 + str(i)
                    sent = sent.replace(cur_attr, template_config[attr2 + '_neutral'][i])

            for target in target_words:
                sample = sent.replace(template_config['target'], target)
                test_data.append(sample)
                attr_label.append(attr)
                target_label.append(target)

    return test_data, attr_label, target_label


def data_model_bias_corr(stat_path, df_task):
    # compute r2
    df = pd.read_csv(stat_path)
    all_data_bias = []
    all_pretrain_bias = []

    if not df.shape == df_task.shape:
        print("shape mismatch for logged training biases and pretrain biases")
        print(df.shape, "vs. ", df_task.shape)
        print("pre-training statistics:")
        print(df)
        print("biases after training:")
        print(df_task)

    for i in range(df.shape[1]):
        if df.columns[i] == 'Unnamed: 0':
            continue
        data_bias = list(df.loc[:, df.columns[i]])
        pretrain_bias = list(df_task.loc[:, df.columns[i]])
        all_data_bias += data_bias
        all_pretrain_bias += pretrain_bias

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(all_data_bias,
                                                                         all_pretrain_bias)
    print("data - eval r2: ", r_value, "(", p_value, ")")
    return r_value, p_value


def run(config, min_iter=0, max_iter=-1):

    print("load templates and protected attributes...")
    with open(config['template_file'], 'r') as f:
        tmp = yaml.safe_load(f)

    target_domain = tmp['target']
    target_words = tmp[target_domain]
    protected_attributes = tmp['protected_attr']

    protected_groups = {}
    group_attr = []
    for attr in protected_attributes:
        protected_groups.update({attr: tmp[attr][0]})
        for i in range(len(tmp[attr])):
            group_attr += tmp[attr][i]

    print("check occurence of protected groups in the training and test templates...")
    attribute_stats = {}
    for attr in protected_attributes:
        attribute_stats.update({attr: {'train': 0, 'test': 0}})

    n_train = len(tmp['templates_train'])
    n_test = len(tmp['templates_test'])

    for temp in tmp['templates_train']:
        for attr in protected_attributes:
            if attr in temp:
                attribute_stats[attr]['train'] += 1

    for temp in tmp['templates_test']:
        for attr in protected_attributes:
            if attr in temp:
                attribute_stats[attr]['test'] += 1

    for attr, entry in attribute_stats.items():
        entry['train'] /= n_train
        entry['test'] /= n_test

    print(attribute_stats)

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
                print("handling iteration ", iter_id, "with params:")
                print("minP:", minP, "maxP: ", maxP, "iteration: ", it)
                iter_results = config['results_dir'] + '/' + str(iter_id)
                if not os.path.exists(iter_results):
                    os.makedirs(iter_results)
                iter_lookup.update({iter_id: (minP, maxP, it)})
                model_path = iter_results+'/model'
                data_path = iter_results+'/train_data.pickle'
                stat_path = iter_results+'/train_data_stats.csv'
                model_bias_path = iter_results + "/task_res.csv"
                iter_config = {'minP': minP, 'maxP': maxP, 'iteration': it, 'base_config': log_config,
                               'model': model_path, 'data': data_path, 'stat': stat_path}
                with open(iter_results+'/config.yaml', 'w') as file:
                    yaml.dump(iter_config, file)

                X_train = []
                y_train = []

                target_group_occ = {}
                for attr in protected_attributes:
                    for group in tmp[attr][0]:
                        target_group_occ.update({group: {}})

                if not os.path.isfile(data_path):
                    print("create dataset from templates with minP and maxP parameters and save it...")

                    if config['objective'] == 'MLM':
                        data, X_train, y_train = create_masked_dataset(tmp, probs_by_attr, target_words, 'templates_train')

                    elif config['objective'] == 'MLM_lazy':
                        data, _, X_train = create_masked_dataset(tmp, probs_by_attr, target_words, 'templates_train')

                    else:
                        print("objective not supported, please select 'MLM' or 'MLM_lazy'")
                        exit(1)

                    data_save = {'samples': X_train, 'labels': y_train, 'epochs': config['epochs']}

                    with open(data_path, "wb") as handler:
                        pickle.dump(data_save, handler)

                    print("log co-occurence of target words and protected groups...")
                    for group in target_group_occ.keys():
                        for target in target_words:
                            target_group_occ[group].update({target: 0})

                    for sample in data:
                        for attr in protected_attributes:
                            if sample[attr] > -1:
                                target_group_occ[protected_groups[attr][sample[attr]]][sample['target']] += 1

                    df = pd.DataFrame(data=target_group_occ)

                    # normalize per attribute
                    for attr in protected_attributes:
                        # overall occurence of this attribute (equal for all target words)
                        sel = df.loc[target_words[0], tmp[attr][0]]
                        sum = np.sum(sel)
                        # normalize
                        df.loc[:, tmp[attr][0]] /= sum

                    print(df)
                    df.to_csv(stat_path)

                else:
                    print("load training data from "+data_path)
                    with open(data_path, "rb") as handler:
                        data_save = pickle.load(handler)

                    df = pd.read_csv(stat_path)

                    X_train = data_save['samples']
                    y_train = data_save['labels']

                training_iterations = 5
                saved_r = 0
                if os.path.isdir(model_path):
                    print("model is already trained, load checkpoint")
                    if 'MLM' in config['objective']:
                        bert = BertHuggingfaceMLM(model_name=config['pretrained_model'],
                                                  batch_size=config['batch_size'])
                    else:
                        bert = BertHuggingface(model_name=config['pretrained_model'], batch_size=config['batch_size'],
                                               num_labels=2)
                    bert.load(model_path)
                    test_data = data_save['test_data']
                    target_label = data_save['target_label']
                    attr_label = data_save['attr_label']
                    saved_r = data_save['baseline_r2']
                    if 'iter_left' in data_save.keys():
                        training_iterations = data_save['iter_left']
                        print("need to resume training with", training_iterations, "iterations left")
                if training_iterations > 0:
                    print("retrain BERT with ", len(X_train), " training samples for ", config['epochs'], " epochs")

                    r_value = saved_r
                    last_r_value = -1
                    it = 0
                    while r_value < 0.85 and it < training_iterations:
                        if config['objective'] == 'MLM_lazy':
                            bert = BertHuggingfaceMLM(model_name=config['pretrained_model'],
                                                      batch_size=config['batch_size'])
                            losses = bert.retrain(X_train, X_train, epochs=config['epochs'], insert_masks=True)
                            df_task, probs, test_data, attr_label, target_label = unmasking_bias_multi_attr(bert, tmp,
                                                                                                            target_words,
                                                                                                            target_group_occ.keys())
                        elif config['objective'] == 'MLM':
                            bert = BertHuggingfaceMLM(model_name=config['pretrained_model'],
                                                      batch_size=config['batch_size'])
                            losses = bert.retrain(X_train, y_train, epochs=config['epochs'])
                            df_task, probs, test_data, attr_label, target_label = unmasking_bias_multi_attr(bert, tmp,
                                                                                                            target_words,
                                                                                                            target_group_occ.keys())
                        else:
                            print("objective not supported, please select 'MLM' or 'MLM_lazy'")
                            exit(1)

                        r_value, p_value = data_model_bias_corr(stat_path, df_task)

                        it += 1
                        if r_value > last_r_value:
                            print("save model with r_value of ", r_value)
                            bert.save(model_path)
                            df_task.to_csv(model_bias_path)

                            data_save['test_probs'] = probs
                            data_save['loss'] = losses
                            data_save['iter_left'] = training_iterations-it
                            data_save['baseline_r2'] = r_value
                            print("iterations left: ", data_save['iter_left'])
                            data_save['test_data'] = test_data
                            data_save['target_label'] = target_label
                            data_save['attr_label'] = attr_label
                            with open(data_path, "wb") as handler:
                                pickle.dump(data_save, handler)

                        last_r_value = r_value
                    print("done with training, final r_value: ", r_value)
                    data_save['iter_left'] = 0
                    with open(data_path, "wb") as handler:
                        pickle.dump(data_save, handler)

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

                data_save['overall_biases'] = df_overall
                for score in score_names:
                    data_save[score+'_bias'] = dataframes[score]
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
