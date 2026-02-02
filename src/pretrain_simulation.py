import numpy as np
import os
import pandas as pd
import pickle
import scipy
import yaml
import getopt
import sys
import re
import math

from tqdm import tqdm

import torch
#from transformers import pipeline
from utils import create_bias_distribution, check_config, check_attribute_occurence, create_masked_dataset, templates_to_train_samples, templates_to_eval_samples
from embedding import BertHuggingfaceMLM #, BertHuggingface
from geometrical_bias import SAME, WEAT, GeneralizedWEAT, DirectBias, RIPA, MAC
from lipstick_bias import BiasGroupTest, NeighborTest, ClusterTest, ClassificationTest
from unmasking_bias import PLLBias


DEBUG = False

class DatasetForTransformer(torch.utils.data.Dataset):

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['index'] = idx
        return item

    def __len__(self):
        return len(self.encodings.input_ids)


def forward_mlm(bert, texts: list[str], attr_terms: list[str], verbose=False):
    emb_dim = bert.model.config.hidden_size
    max_length = 512
    
    vocab_ids = [bert.tokenizer.get_vocab().get(word) for word in attr_terms]
    inputs = bert.tokenizer(texts, return_tensors='pt', max_length=max_length, truncation=True,
                            padding='max_length')
    dataset = DatasetForTransformer(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=bert.batch_size, shuffle=False)

    output_emb = np.zeros((len(texts), emb_dim))
    output_prob = np.zeros((len(texts), len(vocab_ids)))
    for batch in tqdm(loader, leave=True):
        if torch.cuda.is_available():
            for key in batch.keys():
                if key != 'index':
                    batch[key] = batch[key].to('cuda')

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        indices = batch['index']

        out = bert.model(input_ids, attention_mask=attention_mask)
        logits = out.logits # shape: batch_size, tokens, vocab size
        token_emb = out.hidden_states[-1] # shape: batch size, tokens, emb dim

        # mask the [mask] tokens and assert exactly one mask token per sample
        mask = input_ids == bert.tokenizer.mask_token_id
        row_counts = mask.sum(dim=1)
        
        if not torch.all(row_counts == 1):
            for index in indices:
                print(texts[index])
        assert torch.all(row_counts == 1)
        
        # get mask token indices
        masked_indices = torch.nonzero(mask, as_tuple=False)
        batch_ids = masked_indices[:,0]
        token_ids = masked_indices[:,1]

        # get mask token probabilities for selected targets
        masked_logits = logits[batch_ids, token_ids, :] 
        probs = masked_logits.softmax(dim=-1)
        target_probs = probs[:, vocab_ids]
        
        # get mask token embeddings
        mask_emb = token_emb[batch_ids, token_ids, :]

        output_emb[indices.numpy()] = mask_emb.to('cpu').detach().numpy()
        output_prob[indices.numpy()] = target_probs.to('cpu').detach().numpy()

        input_ids = input_ids.to('cpu')
        attention_mask = attention_mask.to('cpu')
        logits = logits.to('cpu')
        token_emb = token_emb.to('cpu')

        del input_ids
        del attention_mask
        del logits
        del token_emb
        torch.cuda.empty_cache()
    return output_emb, output_prob


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


def compute_semantic_bias_group_scores(emb_eval: np.ndarray, emb_def: list[np.ndarray], targets: list[str], protected_groups: list[str], target_stat_df: pd.DataFrame):
    SEM_SCORES_GROUP = [WEAT(), GeneralizedWEAT(), ClusterTest(), ClassificationTest(), NeighborTest(k=100), WEAT(), GeneralizedWEAT(), ClusterTest(), ClassificationTest(), NeighborTest(k=100)]
    SEM_SCORE_NAMES_GROUP = ["WEAT", "GWEAT", "cluster", "classification", "neighbor", "WEAT_i", "GWEAT_i", "cluster_i", "classification_i", "neighbor_i"]

    # these scores require groups of target words for each protected group, assign based on co-occurence in the data:
    group_label_per_target, group_label_per_target_i = partition_target_groups(protected_groups, target_stat_df)

    sample_bias_scores = {}
    agg_bias_scores = {}
    for score in SEM_SCORE_NAMES_GROUP:
        agg_bias_scores.update({score: {}})

        if 'WEAT' in score: # only this one is defined for sample bias
            sample_bias_scores.update({score: {}})
            for target in targets:
                sample_bias_scores[score].update({target: []})

    y = np.asarray([group_label_per_target[target] for target in targets]) 
    y_i = np.asarray([group_label_per_target_i[target] for target in targets])

    emb_lists = []
    n_groups = len(emb_def)
    for c in range(n_groups):
        c_emb = [emb_eval[i] for i in range(len(emb_eval)) if y[i] == c]
        emb_lists.append(c_emb)
        print("emb list for group", protected_groups[c], "has len", len(c_emb))
        assert len(c_emb) > 0, "got zero targets for group: "+protected_groups[c]

    for idx, score in enumerate(SEM_SCORES_GROUP):
        score_name = SEM_SCORE_NAMES_GROUP[idx]

        # adapt score name and labels for ideal / noisy label experiments
        score_name_short = score_name[:-2] if "_i" in score_name else score_name
        cur_y = y_i if "_i" in score_name else y

        if (score_name_short in ["WEAT", "cluster"] and len(emb_def) > 2):
            # skip binary scores in non-binary settings
            agg_bias_scores[score_name] = math.nan

            continue

        if score_name_short not in ['cluster', 'classification', 'neighbor']:
            score.define_bias_space(emb_def)

        # individual bias score (only defined for WEAT)
        if score_name_short == "WEAT":
            for i, target in enumerate(targets):
                sample_bias_scores[score_name][target].append(score.individual_bias(emb_eval[i]))

        # overall bias scores (cosine scores and lipstick tests
        if score_name_short in ["WEAT", "GWEAT"]:
            agg_bias_scores[score_name] = score.group_bias(emb_lists)
        elif score_name_short == "cluster":
            agg_bias_scores[score_name] = score.cluster_test_with_labels(emb_eval, cur_y)
        elif score_name_short == "classification":
            agg_bias_scores[score_name] = np.mean(score.classification_test_with_labels(emb_eval, cur_y))
        else:  # score_name_short == "neighbor"
            agg_bias_scores[score_name] = np.mean(score.bias_by_neighbor(emb_lists))

    return agg_bias_scores, sample_bias_scores
    
        
def compute_semantic_bias_mean_scores(emb_eval: np.ndarray, emb_def: list[np.ndarray], targets: list[str], protected_groups: list[str]):
    # expecting defining embeddings, test embeddings and corresponding target labels for just one attribute
    SEM_SCORES = [SAME(), MAC(), DirectBias(), RIPA()]
    SEM_SCORE_NAMES = ["SAME", "MAC", "DirectBias", "RIPA"]

    sample_bias_scores = {}
    agg_bias_scores = {}
    for score in SEM_SCORE_NAMES:
        agg_bias_scores.update({score: {}})
        sample_bias_scores.update({score: {}})
        for target in targets:
            sample_bias_scores[score].update({target: []})

    for idx, score in enumerate(SEM_SCORES):
        score_name = SEM_SCORE_NAMES[idx]
        score_name_short = score_name[:-2] if "_i" in score_name else score_name

        score.define_bias_space(emb_def)

        # individual bias scores (all scores here implement this)
        for i, target in enumerate(targets):
            if score_name == 'SAME' and len(emb_def) == 2:
                sample_bias_scores[score_name][target].append(score.signed_individual_bias(emb_eval[i]))
            else:
                sample_bias_scores[score_name][target].append(score.individual_bias(emb_eval[i]))

        # aggregate bias (here mean of sample bias)
        agg_bias_scores[score_name] = score.mean_individual_bias(emb_eval)
    
    return agg_bias_scores, sample_bias_scores
    

def compute_semantic_bias(emb_eval: np.ndarray, emb_def: list[np.ndarray], targets: list[str], protected_groups: list[str], target_stat_df: pd.DataFrame):
    # compute mean and group bias scores
    agg_mean, sample_mean = compute_semantic_bias_mean_scores(emb_eval, emb_def, targets, protected_groups)
    agg_groups, sample_groups = compute_semantic_bias_group_scores(emb_eval, emb_def, targets, protected_groups, target_stat_df)
    
    # merge results
    sample_bias_scores = {**sample_mean, **sample_groups}
    agg_bias_scores = {**agg_mean, **agg_groups}

    # post-process sample bias results:
    #   - compute mean sample bias per target word (over templates)
    #   - set missing values
    target_bias = {} # one dataframe with sample biases per score
    for score, score_res in sample_bias_scores.items():
        mean_bias_by_target = {target: np.mean(target_res) if len(target_res) > 0 else math.nan for target, target_res in score_res.items()}
        target_bias[score] = mean_bias_by_target

    return agg_bias_scores, target_bias


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
                                                target_words, config)
        data_test = templates_to_eval_samples(tokenizer, template_config, target_words)
        data_save = {'train': data_train, 'test': data_test, 'epochs': config['epochs']}

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
            # normalize
            df_data_stats.loc[group, :] /= sel_sum

        df_data_stats.to_csv(stat_path, index_label='groups')
    else:
        print("load training data from "+data_path)
        with open(data_path, "rb") as handler:
            data_save = pickle.load(handler)

        df_data_stats = pd.read_csv(stat_path, index_col='groups')

    #print(df_data_stats)

    return data_save, df_data_stats


def forward_test_data(bert: BertHuggingfaceMLM, data_test: list, protected_attributes: list):
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
        mask_emb, mask_prob = forward_mlm(bert, texts=masked_sentences, attr_terms=attr_choices)

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


def evaluate_semantic_biases(def_emb: dict, emb_per_attr: dict, targets_per_attr: dict, protected_attributes: list, protected_groups: dict, df_data_stats: pd.DataFrame):
    # compute bias scores per attribute
    scores_agg = {}
    scores_target = {}
    scores_target_pair = {}
    all_unmask_probs = []
    for attr in protected_attributes:
        print("compute bias for ", attr)
        agg_bias_scores, target_bias = compute_semantic_bias(emb_per_attr[attr], def_emb[attr], targets_per_attr[attr], protected_groups[attr], df_data_stats)
        scores_agg[attr] = agg_bias_scores
        scores_target[attr] = target_bias

    score_names = scores_target[protected_attributes[0]].keys()

    # aggregated bias scores (per score and attribute)
    df_agg = pd.DataFrame(data=scores_agg)

    # target-wise bias scores, one dataframe per score (per target and attribute)
    target_dfs = {}
    for score in score_names:
        score_res = {attr: res[score] for attr, res in scores_target.items()}
        #print(score_res)
        target_dfs[score] = pd.DataFrame(data=score_res)

    return df_agg, target_dfs


def run(config, min_iter=0, max_iter=-1):

    print("load templates and protected attributes...")
    with open(config['template_file'], 'r') as f:
        template_config = yaml.safe_load(f)

    target_domain = template_config['target']
    target_words = template_config[target_domain]
    if DEBUG:
        target_words = target_words[:20]
    protected_attributes = template_config['protected_attr']

    protected_groups = {}
    group_attr = []
    for attr in protected_attributes:
        protected_groups.update({attr: template_config[attr][0]})
        for i in range(len(template_config[attr])):
            group_attr += template_config[attr][i]

    #print(protected_groups)

    check_attribute_occurence(template_config)

    # save config and create dir for all artifacts
    if not os.path.isdir(config['results_dir']):
        os.makedirs(config['results_dir'])
    log_config = config['results_dir']+'/config.yaml'

    with open(log_config, 'w') as file:
        yaml.dump(config, file)

    print("minP choices: ", config['minP'])
    print("maxP choices: ", config['maxP'])
    print("iterations: ", config['iterations'])
    exp_id = -1 # experiment iteration (one combination of minP, maxP and it - saved by this ID)
    for minP in config['minP']:
        for maxP in config['maxP']:
            # set experiment ID (for one set of minP, maxP) and check if this should be run
            exp_id += 1
            if exp_id < min_iter:
                continue
            if (exp_id > max_iter and not max_iter == -1):
                print("finished experiment with ID max_iter, stop now")
                return

            print("handling experiment ", exp_id, "with params:")
            print("minP:", minP, "maxP: ", maxP)

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

                # load pretrained model (need tokenizer to create dataset)
                bert = BertHuggingfaceMLM(model_name=config['pretrained_model'], batch_size=config['batch_size'])


                # create or load the dataset
                data_save, df_data_stats = create_dataset(data_path, stat_path, bert.tokenizer, template_config, probs_by_attr, target_words, config, 
                                                         protected_attributes, protected_groups)
                data_test = data_save['test']
                data_train = data_save['train']
                X_train = [sample['masked_sentence'] for sample in data_train]
                y_train = [sample['sentence'] for sample in data_train]
                
                # training (load from checkpoint if possible, try multiple iterations until good r-value for unmasking probs)
                checkpoint_exists = os.path.isdir(model_path)
                training_iterations_left = config['max_retries']
                r_value = 0
                last_r_value = -1
                if 'baseline_r2' in data_save.keys() and os.path.isdir(model_path):
                    print("found checkpoint for this model, check if further training is necessary")
                    r_value = data_save['baseline_r2']
                    last_r_value = r_value
                    training_iterations_left = data_save['iter_left']
                    if training_iterations_left > 0 and r_value < config['target_r_value']:
                        print("retraining necessary with %i iterations left" % training_iterations_left)

                it = 0
                while r_value < config['target_r_value'] and it < training_iterations_left:
                    print("retrain BERT with ", len(data_train), " training samples for ", config['epochs'], " epochs")

                    # reset Bert to pretrained state
                    bert = BertHuggingfaceMLM(model_name=config['pretrained_model'], batch_size=config['batch_size'], lr=float(config['learning_rate']))

                    # train
                    print("retrain...")
                    losses = bert.retrain(X_train, y_train, epochs=config['epochs'])

                    # evaluate
                    print("evaluate...")
            
                    # pass test data through model to obtain unmasking probabilities and embeddings
                    emb_per_attr, prob_per_attr, targets_per_attr = forward_test_data(bert, data_test, protected_attributes)
                    
                    # compute unmasking bias and verify if unmasking probs align with the data distribution
                    corr_res, unmask_scores_agg, unmask_scores_target, df_unmask = evaluate_unmasking(emb_per_attr, prob_per_attr, targets_per_attr, protected_attributes, template_config, df_data_stats)
                    r_value = corr_res['r']

                    it += 1
                    if r_value > last_r_value:
                        print("save model with r_value of ", r_value)
                        bert.save(model_path)
                        df_unmask.to_csv(model_bias_path)

                        data_save['emb_per_attr'] = emb_per_attr
                        data_save['prob_per_attr'] = prob_per_attr
                        data_save['targets_per_attr'] = targets_per_attr
                        data_save['baseline_r2'] = r_value
                        data_save['iter_left'] = training_iterations_left-it
                        data_save['corr_res'] = corr_res
                        data_save['unmask_score_agg'] = unmask_scores_agg
                        data_save['unmask_bias'] = unmask_scores_target

                        with open(data_path, "wb") as handler:
                            pickle.dump(data_save, handler)

                    last_r_value = r_value

                # either achieved proper r-value or max number of iterations reached
                print("done with training, got final r_value: ", r_value)

                # set remaining iteration to 0
                data_save['iter_left'] = 0
                with open(data_path, "wb") as handler:
                    pickle.dump(data_save, handler)

                # re-load the model from checkpoint and the unmasking artifacts (in case the last run might hasn't been the best one)
                bert.load(model_path)
                emb_per_attr = data_save['emb_per_attr']
                targets_per_attr = data_save['targets_per_attr']
                unmask_scores_agg = data_save['unmask_score_agg']
                unmask_scores_target = data_save['unmask_bias']

                # evaluate semantic bias score, save final results
                def_emb = create_defining_embeddings_from_templates(bert, template_config)
                for k, v in def_emb.items():
                    v2 = list(zip(*v))
                    def_emb[k] = []
                    for tup in v2:
                        def_emb[k].append(np.asarray(tup))

                df_agg, target_dfs = evaluate_semantic_biases(def_emb, emb_per_attr, targets_per_attr, protected_attributes, protected_groups, df_data_stats)
                
                # merge dataframe with aggregated bises (semantic bias + unmask)
                df_agg = pd.concat([df_agg, pd.DataFrame(unmask_scores_agg, index=['unmask'])])
                data_save['agg_bias'] = df_agg
                for score, df in target_dfs.items():
                    data_save[score+'_bias'] = df
                print("data save keys:")
                print(data_save.keys())

                with open(data_path, "wb") as handler:
                    print("save data")
                    pickle.dump(data_save, handler)
    
    print("done with experiments")



def main(argv):
    config_path = ''
    min_iter = 0
    max_iter = -1
    try:
        opts, args = getopt.getopt(argv, "hc:", ["config=", "min=", "max="])
    except getopt.GetoptError:
        print('pretrain_simulation.py -c <config>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('pretrain_simulation.py -c <config>')
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
