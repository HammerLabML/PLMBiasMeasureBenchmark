import numpy as np
import math
import os
import pandas as pd
from operator import itemgetter
import pickle
from tqdm import tqdm
import scipy
import random
import json

import difflib
import string

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
import seaborn as sns

import torch
from torch import Tensor
import datasets
from datasets import load_dataset
from embedding import BertHuggingfaceMLM

from torch.utils.data import DataLoader, TensorDataset

from geometrical_bias import SAME, WEAT, GeneralizedWEAT, DirectBias, RIPA, MAC, normalize, cossim, EmbSetList, EmbSet, GeometricBias
from unmasking_bias import PLLBias

from utils import CLFHead, SimpleCLFHead, MLMPipeline, CustomModel, CrowSPairsDataset, JigsawDataset, BiosDataset


with open('data/protected_groups.json', 'r') as f:
    pg_config = json.load(f)
    
with open('configs/mlm_exp.json', 'r') as f:
    exp_config = json.load(f)
    
with open(exp_config['batch_size_lookup'], 'r') as f:
    batch_size_lookup = json.load(f)
    
groups_by_bias_types = pg_config['groups_by_bias_types']
terms_by_groups = pg_config['terms_by_groups']

cosine_scores = {'SAME': SAME, 'WEAT': WEAT, 'gWEAT': GeneralizedWEAT, 'DirectBias': DirectBias, 'MAC': MAC}


def run_mlm_experiments(exp_config: dict):
    save_file = exp_config['save_file']
    if os.path.isfile(save_file):
        print("load previous results...")
        with open(save_file, 'rb') as handle:
            res = pickle.load(handle)
            exp_parameters = res['params']
            results = res['results']
    else:        
        exp_parameters = []
        results = []
        for bt in exp_config['bias_types']:
            # mlm experiments
            for mlm in exp_config['mlm']:
                params = {key: exp_config[key] for key in ['bias_scores', 'debias']}
                params.update({'bias_type': bt, 'mlm': mlm})
                
                if exp_config['debias']:
                    # one without debias
                    params = {'bias_type': bt, 'mlm': mlm, 'bias_scores': exp_config['bias_scores'], 'debias': False, 'debias_k': 0}
                    exp_parameters.append(params)
                    for k in exp_config['debias_k']:
                        params = {'bias_type': bt, 'mlm': mlm, 'bias_scores': exp_config['bias_scores'], 'debias': False, 'debias_k': k}
                        exp_parameters.append(params)
                                   
                exp_parameters.append(params)
    print(exp_parameters)
                            
    # load the datasets
    csp_dataset = CrowSPairsDataset(groups_by_bias_types, terms_by_groups)
    
    for i, params in enumerate(exp_parameters):
        if i < len(results):
            print("skip experiment", i, "which is part of the last checkpoint")
            continue
            
        print("run experiment", i, "of", len(exp_parameters), "with parameters:")
        print(params)

        #if 'mlm' in params:
        # MLM experiment with CrowS-Pairs
        if not csp_dataset.sel_attributes(params['bias_type']):
            print("skip mlm experiment for bias type", params['bias_type'])
            continue

        n_groups = len(csp_dataset.sel_groups)
        
        model_name = params['mlm']
        if not model_name in batch_size_lookup.keys():
            print("batch size for model", model_name, "not specified, use 1")
            batch_size = 1
        else:
            batch_size = batch_size_lookup[model_name]
            
        attributes = [terms_by_groups[group] for group in groups_by_bias_types[params['bias_type']]]
        
        debias_ks = []
        if params['debias']:
            debias_ks = params['debias_k']
        pipeline = MLMPipeline(parameters={'debias': params['debias'], 'debias_k': debias_ks, 'batch_size': batch_size}, model_name=model_name)
        if params['debias']:
            pipeline.fit_debias(attributes)

        csp_dataset.compute_group_bias(pipeline.model_name, pipeline.compare_sentence_likelihood)
        csp_dataset.compute_individual_bias(pipeline.model_name, pipeline.compare_sentence_likelihood)
        cur_result = {'id': i, 'extrinsic_individual': csp_dataset.individual_biases, 'extrinsic': csp_dataset.bias_score}

        attr_emb = [pipeline.embed(attr, average='mean') for attr in attributes]

        targets, group_label = csp_dataset.get_neutral_samples_by_masking(pipeline.tokenizer)
        assert len(set(group_label)) == n_groups

        target_emb = pipeline.embed(targets, average='mean')
        if params['debias']:
            target_emb2 = pipeline.debiaser.predict(np.asarray(target_emb), pipeline.debias_k)
            print(target_emb2 == target_emb)
            target_emb = target_emb2

        # sorted by stereotypical group
        target_emb_per_group = []
        for group in range(max(group_label)+1):
            target_emb_per_group.append([target_emb[i] for i in range(len(group_label)) if group_label[i] == group])

        for score in params['bias_scores']:
            if score == 'WEAT' and n_groups > 2:
                cur_result.update({score: math.nan})
                continue

            cur_score = cosine_scores[score]()
            cur_score.define_bias_space(attr_emb)

            if not score == 'gWEAT':
                individual_biases = [cur_score.individual_bias(target) for target in target_emb]
                cur_result.update({score+'_individual': individual_biases})

            if score in ['WEAT', 'gWEAT']:
                bias = cur_score.group_bias(target_emb_per_group)
            else:
                # SAME, DirectBias, MAC
                bias = cur_score.mean_individual_bias(target_emb)

            cur_result.update({score: bias})
        results.append(cur_result)
        
        if i+1 < len(exp_parameters) and (exp_parameters[i+1]['debias'] != params['debias'] or exp_parameters[i+1]['debias_k'] != params['debias_k']):
            print("reset PLL results")
            csp_dataset.pll_cur_bias_type = None # reset to make sure that PLL will be computed in next iteration
        
        with open(save_file, 'wb') as handle:
            pickle.dump({'params': exp_parameters, 'results': results}, handle)
        print()
        
        # remove model from GPU
        pipeline.embedder.to('cpu')
        pipeline.head.to('cpu')
        del pipeline
        torch.cuda.empty_cache()
        
    with open(save_file, 'wb') as handle:
        pickle.dump({'params': exp_parameters, 'results': results}, handle)

run_mlm_experiments(exp_config)
print('done')