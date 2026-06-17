import os
import getopt
import sys
import pickle
import yaml

import numpy as np
import math
import pandas as pd
import scipy

import re
import random
from tqdm import tqdm

import torch
from utils import (create_bias_distribution, check_config, check_attribute_occurence, create_masked_dataset, templates_to_train_samples, templates_to_eval_samples, 
                   evaluate_mlm, forward_mlm_for_bias_eval, load_wikitext, mask_texts)

from embedding import BertHuggingfaceMLM
from unmasking_bias import PLLBias

DEBUG = False

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


def compute_unmasking_bias(raw_probs: np.ndarray, targets: list[str], protected_groups: list[str]):
    assert raw_probs.ndim == 2, "expected 2D array with unmasking probabilities of shape: (n_sentences*n_target, n_groups)"
    assert len(targets) == raw_probs.shape[0], "expected probability array and target list to share the same length (n_sentences*n_target)"

    targets = np.array(targets) # target labels per sample
    target_words = np.unique(targets)
    jsd_per_target = {target: None for target in target_words}
    mean_prob_by_target_group = {target: {} for target in target_words}
    for target in target_words:
        # compute per target
        target_probs = raw_probs[targets == target]
        sums = np.sum(target_probs, axis=1)
        rel_probs = target_probs / sums[:, None]
        mean_prob = np.mean(rel_probs, axis=0)

        mean_prob_by_target_group[target] = {grp: mean_prob[i] for i, grp in enumerate(protected_groups)}

        dist_equal = np.ones(mean_prob.shape)/mean_prob.shape[0]
        jsd = scipy.spatial.distance.jensenshannon(mean_prob, dist_equal)
        jsd_per_target[target] = jsd

    mean_jsd = np.mean([jsd for target, jsd in jsd_per_target.items()])

    return jsd_per_target, mean_jsd, mean_prob_by_target_group


def partition_target_groups(protected_groups: list[str], target_stat_df: pd.DataFrame):
    # protected groups should only include the groups for the current attribute
    group_label_per_target = {}  # labels with some noise (assuming biases in the data do not correspond exactly to biases in society/ assumptions of the user)
    group_label_per_target_i = {}  # ideal labels (exact knowledge of biases in the data)
    
    #print(target_stat_df)
    #print(protected_groups)
    probs = target_stat_df.loc[protected_groups, :]

    mu, sigma = 0, 0.3
    noise = np.random.normal(mu, sigma, probs.shape)
    probs_noise = probs.to_numpy()+noise
    group_label = np.argmax(probs_noise, axis=0)
    group_label_i = np.argmax(probs.to_numpy(), axis=0)
    #print("ideal vs. noisy group labels:")
    #print(group_label_i)
    #print(group_label)

    target_words = target_stat_df.columns
    #print(target_words)
    for i, target in enumerate(target_words):
        group_label_per_target.update({target: group_label[i]})
        group_label_per_target_i.update({target: group_label[i]})
    return group_label_per_target, group_label_per_target_i


def data_model_bias_corr(df_data, df_task):
    # compute r2 between the probability distribution of the training data and the unmasking probabilities
    #df = pd.read_csv(stat_path)
    all_data_bias = []
    all_pretrain_bias = []

    if not df_data.shape == df_task.shape:
        # this happens if one dataframe was loaded from csv and contains an 'unnamed: 0' column
        # with the targets
        print("shape mismatch for logged training biases and pretrain biases")
        print(df_data.shape, "vs. ", df_task.shape)
        print("pre-training statistics:")
        print(df_data)
        print("biases after training:")
        print(df_task)

    for i in range(df_data.shape[1]):
        data_bias = list(df_data.loc[:, df_data.columns[i]])
        pretrain_bias = list(df_task.loc[:, df_data.columns[i]])
        print(df_data.columns[i], data_bias, pretrain_bias)
        all_data_bias += data_bias
        all_pretrain_bias += pretrain_bias

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(all_data_bias,
                                                                         all_pretrain_bias)
    print("data / unmask prob correlation R: ", r_value, "(p: ", p_value, ")")

    corr_res = {'r': r_value, 'p': p_value, 'slope': slope, 'intercept': intercept, 'std_err': std_err}
    return corr_res


def create_dataset(data_path: str, stat_path: str, tokenizer, template_config: dict, probs_by_attr: dict, target_words: list[str], config: dict, 
                   protected_attributes: list, protected_groups: dict):

    # get all protected groups in a list
    group_list = [group for attr in protected_attributes for group in template_config[attr][0]]

    data_exists = os.path.isfile(data_path)
    if not data_exists:
        print("create dataset from templates with minP and maxP parameters and save it...")
        data_train = templates_to_train_samples(tokenizer, template_config, probs_by_attr,
                                                target_words, config, template_key='templates_train')
        data_val = templates_to_eval_samples(tokenizer, template_config, target_words, template_key='templates_val')
        data_test = templates_to_eval_samples(tokenizer, template_config, target_words, template_key='templates_test')
        data_save = {'train': data_train, 'val': data_val, 'test': data_test, 'epochs': config['epochs']}

        with open(data_path, "wb") as handler:
            pickle.dump(data_save, handler)

        print("log co-occurence of target words and protected groups...")
        target_group_occ = {}
        for target in target_words:
            target_group_occ[target] = {group: 0 for group in group_list}

        for sample in data_train:
            for attr in protected_attributes:
                if sample[attr] > -1:  # group id ( > -1 if attribute exists)
                    target_group_occ[sample['target']][protected_groups[attr][sample[attr]]] += 1

        df_data_stats = pd.DataFrame(data=target_group_occ)

        # normalize per group ( -> p(target | group))
        for group in group_list:
            # overall occurence of this group
            sel = df_data_stats.loc[group, :]
            sel_sum = np.sum(sel)
            # normalize (cast to float)
            df_data_stats.loc[group, :] = df_data_stats.loc[group, :].astype(float)
            df_data_stats.loc[group, :] /= sel_sum

        df_data_stats.to_csv(stat_path, index_label='groups')
    else:
        print("load training data from "+data_path)
        with open(data_path, "rb") as handler:
            data_save = pickle.load(handler)

        df_data_stats = pd.read_csv(stat_path, index_col='groups')

    #print(df_data_stats)

    return data_save, df_data_stats


def forward_test_data(bert: BertHuggingfaceMLM, data_test: list, protected_attributes: list, pooling: str):
    print("compute embeddings and unmasking probs...")

    # get embeddings and mask probs, need to run per 'test case' since different terms for protected groups need to be queried from the model
    attr_keys = [sample['attr_key'] for sample in data_test]
    test_cases = list(set(attr_keys))

    emb_per_attr = {attr: [] for attr in protected_attributes}
    prob_per_attr = {attr: [] for attr in protected_attributes}
    targets_per_attr = {attr: [] for attr in protected_attributes}
    #print(test_cases)
    for key in test_cases:
        attr = re.sub(r'\d{1,2}$', '', key)
        #print(key, attr)
        # selection of samples for this specific test case (e.g. GENDER1 or ETHNICITY3)
        cur_samples = [sample for sample in data_test if sample['attr_key'] == key]
        
        # sentence with [MASK] replacing the current protected attribute (others are replaced by neutral terms)
        masked_sentences = [sample['sent_masked_attr'] for sample in cur_samples]
        # terms that could replace the [MASK] token and whose probabilities of the model should be queried
        attr_choices = cur_samples[0]['attr_choices']
        
        # pass through model and get the embeddings and probabilitites of the mask token
        mask_emb, mask_prob = forward_mlm_for_bias_eval(bert, texts=masked_sentences, replace_terms=attr_choices, pooling=pooling)

        emb_per_attr[attr].append(mask_emb)
        prob_per_attr[attr].append(mask_prob)
        for sample in cur_samples:
            targets_per_attr[attr].append(sample['target'])

    for attr in protected_attributes:
        prob_per_attr[attr] = np.vstack(prob_per_attr[attr])
        emb_per_attr[attr] = np.vstack(emb_per_attr[attr])

    return emb_per_attr, prob_per_attr, targets_per_attr


def evaluate_unmasking(emb_per_attr: dict, prob_per_attr: dict, targets_per_attr: dict, protected_attributes: list, template_config: dict, df_data_stats: pd.DataFrame):
    # compute bias scores per attribute
    scores_agg = {}
    scores_target = {}
    scores_target_pair = {}
    all_unmask_probs = []
    for attr in protected_attributes:
        print()
        print("compute unmasking bias for ", attr)
        prob_per_attr[attr] = np.vstack(prob_per_attr[attr])
        emb_per_attr[attr] = np.vstack(emb_per_attr[attr])
        cur_groups = template_config[attr][0]
        unmasking_bias_target, unmasking_bias_agg, unmask_probs = compute_unmasking_bias(prob_per_attr[attr], targets_per_attr[attr], cur_groups)
        
        all_unmask_probs.append(pd.DataFrame(data=unmask_probs))
        scores_agg[attr] = unmasking_bias_agg
        scores_target[attr] = unmasking_bias_target

    # contains the mean probabilities per target and group (to be compared with data statistics)
    df_unmask_prob = pd.concat(all_unmask_probs)
    corr_res = data_model_bias_corr(df_data_stats, df_unmask_prob)

    return corr_res, scores_agg, scores_target, df_unmask_prob


def create_performance_plot(measures: dict[str, list[float]], 
                            errors: dict[str, list[float]], 
                            title: str, 
                            filename: str,
                            width=1000, height=600):
    """
    Creates a Plotly line plot and saves it as PNG.
    
    Args:
        measures: Dict of metric_name -> list of values
        title: Plot title
        filename: Name for output file (without extension)
        width, height: Image dimensions in pixels
    """
    epochs = list(range(len(next(iter(measures.values())))))
    
    fig = go.Figure()
    
    for score_name, scores in measures.items():
        fig.add_trace(go.Scatter(
            x=epochs, 
            y=scores, 
            mode='lines+markers', 
            name=score_name,
            line=dict(width=2)
        ))

        if errros is not None:
            upper_bound = [m + s for m, s in zip(scores, errors[score_name])]
            lower_bound = [m - s for m, s in zip(scores, errors[score_name])]
            
            fig_agg.add_trace(go.Scatter(
                x=epochs_agg + epochs_agg[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor=color_rgb(i), # Custom helper to get alpha color
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name=f"{metric} Std" # Optional: hidden in legend
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Score",
        hovermode="x unified",
        template="plotly_white",
        width=width,
        height=height
    )
    
    try:
        fig.write_image(f"plots/{filename}.png")
        print(f"Saved plot: {filename}.png")
    except Exception as e:
        print(f"Error saving plot {filename}.png: {e}")
        print("Hint: Install kaleido with 'pip install kaleido'")


def evaluate(bert, scores, data_val, data_test, wikitext_data, protected_attributes, config):
    # evaluate unmasking bias on the train (=val) and test set (forward pass to get probabilities then compute bias)
    emb_per_attr, prob_per_attr, targets_per_attr = forward_test_data(bert, data_val, protected_attributes, config['pooling'])
    corr_res_train, unmask_scores_agg, unmask_scores_target, df_unmask = evaluate_unmasking(emb_per_attr, prob_per_attr, targets_per_attr, protected_attributes, template_config, df_data_stats)
    scores['r_train'].append(corr_res_train['r'])

    emb_per_attr, prob_per_attr, targets_per_attr = forward_test_data(bert, data_test, protected_attributes, config['pooling'])
    corr_res_test, unmask_scores_agg, unmask_scores_target, df_unmask = evaluate_unmasking(emb_per_attr, prob_per_attr, targets_per_attr, protected_attributes, template_config, df_data_stats)
    scores['r_test'].append(corr_res_test['r'])
    
    mlm_result = evaluate_mlm(bert, wikitext_data['val'])
    scores['acc'].append(mlm_result['accuracy'])
    scores['ppl'].append(mlm_result['perplexity'])

    return scores


def run(config, min_iter=0, max_iter=-1):

    print("load templates and protected attributes...")
    with open(config['template_file'], 'r') as f:
        template_config = yaml.safe_load(f)

    target_domain = template_config['target']
    target_words = template_config[target_domain]
    if DEBUG:
        target_words = target_words[:10]
    protected_attributes = template_config['protected_attr']

    protected_groups = {}
    group_attr = []
    for attr in protected_attributes:
        protected_groups.update({attr: template_config[attr][0]})
        for i in range(len(template_config[attr])):
            group_attr += template_config[attr][i]

    check_attribute_occurence(template_config)

    # save config and create dir for all artifacts
    if not os.path.isdir(config['results_dir']):
        os.makedirs(config['results_dir'])
    log_config = config['results_dir']+'/config.yaml'

    # check for previous results
    score_names = ['r_test', 'r_train', 'acc', 'ppl']
    results_file = config['results_dir']+'/results.csv'
    all_results = []
    if os.path.isfile(results_file):
        df_results = pd.read_csv(results_file)
        print("got previous results:")
        print(df_results)

        # convert to dict
        for score in score_names:
            df_results[score] = 1 # dummy values to preserve structure
        all_results = df_results.to_dict(orient='records')
    print(all_results)


    with open(log_config, 'w') as file:
        yaml.dump(config, file)

    # load wikitext dataset
    add_wiki_data = config['add_wiki_data']
    wikitext_data = load_wikitext(template_config, version="wikitext-2-raw-v1")

    print("minP choices: ", config['minP'])
    print("maxP choices: ", config['maxP'])
    print("iterations: ", config['iterations'])
    exp_id = -1
    for minP in config['minP']:
        for maxP in config['maxP']:
            exp_id += 1
            print("expID:", exp_id, "minP:", minP, "maxP: ", maxP)

            # check df_results if this has been done
            exists = any(item['minP'] == minP and item['maxP'] == maxP for item in all_results)
            if exists:
                print("got previous results for this, continue")
                continue

            # create bias distribution based on current parameters
            print("create bias distribution...")
            probs_by_attr = {}
            for attr, groups in protected_groups.items():
                n = len(groups)
                res = create_bias_distribution(n, target_words, minP=minP / n, maxP=maxP / n)
                df = pd.DataFrame(data=res)
                probs_by_attr.update({attr: df})

            # run multiple iterations of experiments as specified in config
            for it in range(config['iterations']): # iterations by which one setting (minP, maxP) is repeated
                print("at iteration %i of experiment %i" % (it, exp_id))
                
                # prepare all paths and configs to save artifacts
                iter_results = config['results_dir'] + '/%i_%i' % (exp_id,it)
                if not os.path.exists(iter_results):
                    os.makedirs(iter_results)
                data_path = iter_results+'/data.pickle'
                stat_path = iter_results+'/train_data_stats.csv'

                iter_config = {'minP': minP, 'maxP': maxP, 'iteration': it, 'base_config': log_config,
                               'stat': stat_path}
                config_file = iter_results+'/config.yaml'
                with open(config_file, 'w') as file:
                    yaml.dump(iter_config, file)

                # load pretrained model (need tokenizer to create dataset)
                bert = BertHuggingfaceMLM(model_name=config['pretrained_model'], batch_size=config['batch_size'])

                # create or load the dataset
                data_save, df_data_stats = create_dataset(data_path, stat_path, bert.tokenizer, template_config, probs_by_attr, target_words, config, 
                                                         protected_attributes, protected_groups)
                data_test = data_save['test']
                data_train = data_save['train']
                data_val = data_save['val'] # training templates processed for evaluation
                X_train = [sample['masked_sentence'] for sample in data_train]
                y_train = [sample['sentence'] for sample in data_train]

                if add_wiki_data:
                    # take a sample of the train set (depending on the number of other training samples)
                    n_wiki_samples = len(X_train)
                    if DEBUG:
                        n_wiki_samples = 20
                    wiki_train_sample = random.sample(wikitext_data['train'], n_wiki_samples)

                    # insert masks
                    wiki_train_m = mask_texts(bert, wiki_train_sample, max_length=512, return_tokens=False)

                    # combine training data
                    X_train = X_train + wiki_train_m
                    y_train = y_train + wiki_train_sample

                    # shuffle
                    p = np.random.permutation(len(X_train))
                    X_train, y_train = np.array(X_train)[p].tolist(), np.array(y_train)[p].tolist()
                
                # set up result dict and evaluate once before training
                scores = {'r_test': [], 'r_train': [], 'acc': [], 'ppl': []}
                scores = evaluate(bert, scores, data_val, data_test, wikitext_data, protected_attributes, config)

                # training one epoch at a time and track results
                for ep in range(config['epochs']):
                    print("train (epoch %i)..." % ep)
                    losses = bert.retrain(X_train, y_train, epochs=1)
                    scores = evaluate(bert, scores, data_val, data_test, wikitext_data, protected_attributes, config)
                print(scores)
                
                # plot and collect results
                title_str = f"Performance: minP={minP}, maxP={maxP}, iter={it}"
                file_name_str = iter_results+'/plot'
                create_performance_plot(scores, title=title_str, filename=file_name_str)

                row_data = {'minP': minP, 'maxP': maxP, 'iter': it}
                for metric_name, values in scores.items():
                    # Store values as a comma-separated string or list object
                    # Using object type allows us to convert back to list easily later
                    row_data[metric_name] = values
                row_data['best r'] = np.max(scores['r_test'])
                row_data['best epoch'] = np.argmax(scores['r_test'])  # ordered by epochs anyway and index 0 = eval before training

                print(row_data)
                
                all_results.append(row_data)

            # save results so far
            print("save current results to ", results_file)
            df = pd.DataFrame(all_results)
            print(df)
            df = df.drop(columns=['r_train','r_test','acc','ppl'])
            df.to_csv(results_file)


    # TODO aggregated plot (mean + std over minP,maxP,iter)
    title_str = 'Performance aggregated over minP, maxP, iter'
    agg_plot_filename = config['results_dir']+'/plot_agg'

    # get mean + std of all scores over minP, maxP and iter
    scores_dict = {}
    errors_dict = {}
    for score_name in scores_to_plot.keys():
        scores = np.stack(df[score_name])
        scores_dict[score_name] = np.mean(scores, axis=0)
        errors_dict[score_name] = np.std(scores, axis=0)

    create_performance_plot(scores_dict, errors_dict, title=title_str, filename=agg_plot_filename)  
    
    print("done")



def main(argv):
    config_path = ''
    min_iter = 0
    max_iter = -1
    try:
        opts, args = getopt.getopt(argv, "hc:", ["config=", "min=", "max="])
    except getopt.GetoptError:
        print('evaluate_pretraining_strategies.py -c <config>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('evaluate_pretraining_strategies.py -c <config>')
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
