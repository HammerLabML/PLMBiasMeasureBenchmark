import numpy as np
import math
import os
import pandas as pd
from operator import itemgetter
import pickle
from tqdm import tqdm
import scipy
import random
import yaml

import difflib
import string
import json

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
import seaborn as sns


from sklearn.utils import shuffle

import torch
from torch import Tensor
import datasets
from datasets import load_dataset
from embedding import BertHuggingfaceMLM, BertHuggingface
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from geometrical_bias import SAME, WEAT, GeneralizedWEAT, DirectBias, RIPA, MAC, normalize, cossim, EmbSetList, EmbSet, GeometricBias, cossim
from unmasking_bias import PLLBias

from utils import CLFHead, SimpleCLFHead, CustomModel, JigsawDataset, BiosDataset, DebiasPipeline, upsample_defining_embeddings, WordVectorWrapper
from transformers import AutoModelForMaskedLM, AutoTokenizer
import time


font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)

with open('data/protected_groups.json', 'r') as f:
    pg_config = json.load(f)
    
groups_by_bias_types = pg_config['groups_by_bias_types']
terms_by_groups = pg_config['terms_by_groups']

attributes = [terms_by_groups[group] for group in groups_by_bias_types['gender']]

with open('data/batch_size_lookup_A40.json', 'r') as f:
    batch_size_lookup = json.load(f)


titles = ["architect", "psychologist", "professor", "photographer", "journalist", "attorney", "teacher", "dentist", "*software architect", "*writer", "surgeon", "physician", "nurse", "*researcher"]
titles = [title.replace('*','') for title in titles]
bios_dataset = BiosDataset(n_folds=5, sel_labels=titles, bios_file="../datasets/BIOS_REVIEWED.pkl")
texts = [sample['text'] for sample in bios_dataset.sel_data]
texts_debiased, _, _ = bios_dataset.get_neutral_samples_by_masking(attributes)
y = [sample['label'] for sample in bios_dataset.sel_data]
groups = [sample['group'] for sample in bios_dataset.sel_data]

cosine_scores = {'SAME': SAME(), 'WEAT': WEAT(), 'DirectBias': DirectBias(k=1), 'MAC': MAC()}  
n_titles = len(titles)
n_bios = len(texts)
half_bios = int(n_bios/2)

sample_dist = {title: {'male': 0, 'female': 0} for title in bios_dataset.labels}
for sample in bios_dataset.sel_data:
    for i in range(sample['label'].shape[0]):
        if sample['label'][i] == 1:
            sample_dist[bios_dataset.labels[i]][bios_dataset.sel_groups[sample['group']]] += 1

df = pd.DataFrame(sample_dist)
print("class/gender distribution:")
print(df)
print()

classes_by_majority_group = {'male': [], 'female': []}
for job, dist in sample_dist.items():
    if dist['male'] > dist['female']:
        classes_by_majority_group['male'].append(job)
    else:
        classes_by_majority_group['female'].append(job)

print("classes per majority group: ")
print(classes_by_majority_group)
print()

def compute_bias_scores(attr_emb, target_emb, target_emb_per_group):
    biases_by_scores = {}
    for score_name, score in cosine_scores.items():
        score.define_bias_space(attr_emb)
        
        if score_name in ['WEAT', 'gWEAT']:
            bias = score.group_bias(target_emb_per_group)
        else:
            # SAME, DirectBias, MAC
            bias = score.mean_individual_bias(target_emb)
        biases_by_scores[score_name] = bias
    return biases_by_scores

def get_target_emb_per_group(target_emb, target_label):
    target_emb_per_group = []
    for group in range(max(groups)+1):
        group_name = bios_dataset.sel_groups[group]
        emb = []
        for i in range(len(target_label)):
            for lbl in classes_by_majority_group[group_name]:
                lbl_idx = titles.index(lbl)
                if target_label[i][lbl_idx] == 1:
                    emb.append(target_emb[i])
        target_emb_per_group.append(emb)
    return target_emb_per_group

def test_target_robustness(res_by_model, bias_types=['gender'], models=['bert-base-uncased', 'roberta-base'], n_permutations=100, factors=[0.5, 0.3, 0.1, 0.05, 0.01]):    
    for j, model in enumerate(models):
        if 'done' in res_by_model[model].keys():
            print("skip %s because we already have results", % model)
            continue
        batch_size = 1
        if model in batch_size_lookup.keys():
            batch_size = batch_size_lookup[model]
        lm = BertHuggingface(model_name=model, batch_size=batch_size, num_labels=2)

        for bias_type in bias_types:
            k = len(groups_by_bias_types[bias_type])-1
            print("%s (k=%s)" % (bias_type, k))
            attributes = [terms_by_groups[group] for group in groups_by_bias_types[bias_type]]
    
            attr_emb = np.asarray([lm.embed(attr) for attr in attributes])

            # title bias
            title_emb = lm.embed(titles)
            title_emb_per_group = get_target_emb_per_group(title_emb, np.eye(len(titles)))
            title_bias_scores = compute_bias_scores(attr_emb, title_emb, title_emb_per_group)

            # bios bias (all data)
            bios_emb = lm.embed(texts_debiased)
            target_emb_per_group = get_target_emb_per_group(bios_emb, y)
            bias_scores = compute_bias_scores(attr_emb, bios_emb, target_emb_per_group)

            # compute bias scores on random permuted subsets
            all_ids = list(range(len(texts)))
            for i in tqdm(range(n_permutations)):
                perm_ids = np.random.permutation(all_ids)

                for factor in factors:
                    n_subset = int(factor*n_bios)
                    sel_ids = perm_ids[:n_subset]
    
                    cur_emb = np.asarray([bios_emb[idx] for idx in sel_ids])
                    cur_label = [y[idx] for idx in sel_ids]
                    cur_group = [groups[idx] for idx in sel_ids]
                    
                    cur_emb_per_group = get_target_emb_per_group(cur_emb, cur_label)
    
                    cur_bias_scores = compute_bias_scores(attr_emb, cur_emb, cur_emb_per_group)
                    # TODO compare bias scores to baseline, save deviation
                    for score in cosine_scores:
                        # normalize bc WEAT has larger interval
                        if score == 'WEAT':
                            res_by_model[model][score]['bios_subset'][factor].append(np.abs(cur_bias_scores[score]-bias_scores[score])/4)
                            res_by_model[model][score]['titles/bios'][factor].append(np.abs(cur_bias_scores[score]-title_bias_scores[score])/4)
                        else:
                            res_by_model[model][score]['bios_subset'][factor].append(np.abs(cur_bias_scores[score]-bias_scores[score]))
                            res_by_model[model][score]['titles/bios'][factor].append(np.abs(cur_bias_scores[score]-title_bias_scores[score]))
        res_by_model[model]['done'] = True
            
    return res_by_model

bias_types = ['gender']
factors=[0.01, 0.03, 0.05, 0.1, 0.2]
models=["albert-large-v2", "google/electra-base-generator", "google/electra-large-generator", "bert-base-multilingual-uncased", "GroNLP/hateBERT", "Twitter/twhin-bert-base", "medicalai/ClinicalBERT", "albert-xlarge-v2", "bert-large-uncased-whole-word-masking", "abhi1nandy2/Bible-roberta-base", "distilbert-base-uncased-finetuned-sst-2-english", "gpt2", "openai-gpt", "xlnet-base-cased", "bert-base-uncased", "bert-large-uncased", "distilbert-base-uncased", "roberta-base", "roberta-large", "distilroberta-base", "xlm-roberta-base", "albert-base-v2"]
res_by_model = {model: {score: {'bios_subset': {factor: [] for factor in factors}, 'titles/bios': {factor: [] for factor in factors}} for score in cosine_scores.keys()} for model in models}
res_by_model = test_target_robustness(results, bias_types=bias_types, models=models, n_permutations=50, factors=factors)

with open('robustness_results.pickle','wb') as handle:
    pickle.dump(res_by_model, handle)

score_color_tup = [('SAME', 'blue'), ('DirectBias', 'green'), ('WEAT', 'orange')]
for test in ['bios_subset', 'titles/bios']:
    all_dev = {score: {factor: [] for factor in factors} for score in ['SAME','DirectBias', 'WEAT']}
    for (score, color) in score_color_tup:
        print(score)
        for model in models:
            for factor in factors:
                all_dev[score][factor] += res_by_model[model][score][test][factor]
        print([np.mean(all_dev[score][factor]) for factor in factors])

    if test == 'bios_subset':
        fig, ax = plt.subplots(1)
        for (score, color) in score_color_tup:
            mu_all_dev = np.asarray([np.mean(all_dev[score][factor]) for factor in factors])
            sigma_all_dev = np.asarray([np.std(all_dev[score][factor]) for factor in factors])
            ax.plot(factors, mu_all_dev, lw=2, label=score, color=color)
            ax.fill_between(factors, mu_all_dev+sigma_all_dev, mu_all_dev-sigma_all_dev, facecolor=color, alpha=0.3)
            
        ax.set_title(test)
        ax.legend(loc='upper right')
        ax.set_xlabel('% of dataset')
        ax.set_ylabel('deviation of bias scores')
        ax.grid()
        plt.savefig('plots/target_robustness_bios.png', bbox_inches="tight")
        plt.show()   

print("done")