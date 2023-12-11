import pickle
import numpy as np
import pandas as pd

import torch
import json
import os
import sys
import getopt

from embedding import BertHuggingface
from geometrical_bias import SAME, WEAT, GeneralizedWEAT, DirectBias, MAC, normalize, cossim, EmbSetList, EmbSet, GeometricBias
from utils import CLFHead, SimpleCLFHead, CustomModel, JigsawDataset, BiosDataset, DebiasPipeline, upsample_defining_embeddings, WordVectorWrapper, resample

with open('data/protected_groups.json', 'r') as f:
    pg_config = json.load(f)
    
groups_by_bias_types = pg_config['groups_by_bias_types']
terms_by_groups = pg_config['terms_by_groups']

cosine_scores = {'SAME': SAME, 'WEAT': WEAT, 'gWEAT': GeneralizedWEAT, 'DirectBias': DirectBias, 'MAC': MAC}    


def run_clf_experiments(exp_config: dict):
    with open(exp_config['batch_size_lookup'], 'r') as f:
        batch_size_lookup = json.load(f)
    
    save_dir = exp_config['save_dir']
    save_file = save_dir+'res.pickle'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if os.path.isfile(save_file):
        print("load previous results...")
        with open(save_file, 'rb') as handle:
            res = pickle.load(handle)
            exp_parameters = res['params']
            results = res['results']
            #results_test = res['results_eval'] 
    else:
        exp_parameters = []
        results = []
        #results_test = []

        # prepare parameters for individual experiments
        for bt in exp_config['bias_types']:
            for embedder in exp_config['embedders']:
                for head in exp_config['clf_heads']:
                    for optim in exp_config['clf_optimizer']:
                        for crit in exp_config['clf_criterion']:
                            # one without debias anyway
                            params = {key: exp_config[key] for key in ['bias_scores', 'n_fold', 'batch_size', 'epochs', 'clf_debias']}
                            params.update({'predictions': save_dir+'pred_'+str(len(exp_parameters))+'.pickle'})
                            params.update({'bias_type': bt, 'embedder': embedder, 'head': head, 
                                           'optimizer': optim, 'criterion': crit, 'lr': exp_config['lr'], 'debias': False})
                            exp_parameters.append(params)
                            if exp_config['debias']:
                                for k in exp_config['debias_k']:
                                    params = {key: exp_config[key] for key in ['bias_scores', 'debias', 'n_fold', 'batch_size', 'epochs', 'clf_debias']}
                                    params.update({'predictions': save_dir+'pred_'+str(len(exp_parameters))+'.pickle'})
                                    params.update({'bias_type': bt, 'embedder': embedder, 'head': head, 
                                                   'optimizer': optim, 'criterion': crit, 'lr': exp_config['lr'], 'debias_k': k})
                                    exp_parameters.append(params)        
                            
    # load the datasets    
    bios_merged_file = exp_config['bios_file']
    titles = exp_config['bios_classes']
    n_classes = len(titles)
    bios_dataset = BiosDataset(n_folds=exp_config['n_fold'], sel_labels=titles, bios_file=bios_merged_file)
    print("loaded BIOS dataset with", len(bios_dataset.sel_data), "samples")
    n_groups = len(bios_dataset.sel_groups)

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
    
    
    # run experiments
    for i, params in enumerate(exp_parameters):
        if i < len(results):
            print("skip experiment", i, "which is part of the last checkpoint")
            continue
        
        print()
        print("############################################################################################")
        print("run experiment", i, "of", len(exp_parameters), "with parameters:")
        print(params)
        print("############################################################################################")

        model_name = params['embedder']
        if not model_name in batch_size_lookup.keys():
            print("batch size for model", model_name, "not specified, use 1")
            batch_size = 1
        else:
            batch_size = batch_size_lookup[model_name]
        
        cur_result = {'id': i, 'extrinsic': [], 'extrinsic_individual': [], 'extrinsic_classwise': [], 'subgroup_AUC': [], 'BPSN': [], 'BNSP': [], 
                      'extrinsic_classwise_neutral': [], 'subgroup_AUC_neutral': [], 'BPSN_neutral': [], 'BNSP_neutral': []} # cosine scores on training data
        for score in cosine_scores:
            cur_result.update({score: [], score+'_cf': [], score+'_neutral': [], score+'_individual': [], score+'_classwise': [], score+'_classwise_neutral': []})
            
        print("load model ", model_name)
        is_hugginface_model = True
        if 'fasttext' in model_name or 'word2vec' in model_name or 'glove' in model_name or 'conceptnet' in model_name:
            is_hugginface_model = False
            
        if is_hugginface_model:
            lm = BertHuggingface(model_name=model_name, batch_size=batch_size, num_labels=2)
            emb_size = lm.model.config.hidden_size
        else:
            lm = WordVectorWrapper(model_name)
            emb_size = lm.emb_size
        
        # attributes are independent from data/ fold
        attributes = [terms_by_groups[group] for group in groups_by_bias_types[params['bias_type']]]
        attr_emb = [lm.embed(attr) for attr in attributes]
        
        print("embed all raw bios...")
        target_emb_all = lm.embed([sample['text'].lower() for sample in bios_dataset.sel_data])
        
        print("embed all counterfactual bios...")
        targets_cf, labels_cf, groups_cf = bios_dataset.get_counterfactual_samples(attributes)
        assert len(set(groups_cf)) == n_groups
        cf_emb_all = lm.embed(targets_cf)
        
        print("embed all neutralized bios...")
        targets_neutral, _, _ = bios_dataset.get_neutral_samples_by_masking(attributes)
        neutral_emb_all = lm.embed(targets_cf)
            
        for fold_id in range(params['n_fold']):
            if params['head'] == 'SimpleCLFHead':
                head = SimpleCLFHead(input_size=emb_size, output_size=n_classes)
            elif params['head'] == 'CLFHead':
                head = CLFHead(input_size=emb_size, output_size=n_classes, hidden_size=emb_size)
            else:
                print("invalid clf head: ", params['head'])
                break
            
            # get the training data
            bios_dataset.set_data_split(fold_id)
            
            # new sample distribution and resulting class weights
            mean_n_samples = len(bios_dataset.train_data)/(n_classes*n_groups) # per class
            sample_dist = {title: {'male': 0, 'female': 0} for title in bios_dataset.labels}
            for sample in bios_dataset.train_data:
                for i in range(sample['label'].shape[0]):
                    if sample['label'][i] == 1:
                        sample_dist[bios_dataset.labels[i]][bios_dataset.sel_groups[sample['group']]] += 1

            df = pd.DataFrame(sample_dist)
            print("train data stats for fold ", fold_id)
            print(df)
            class_gender_weights = {g: {lbl: mean_n_samples/df.loc[g,lbl] for lbl in bios_dataset.labels} for g in bios_dataset.sel_groups}
            # for each class: number of negative samples over number of positive samples -> pos weight
            class_weights = [(len(bios_dataset.train_data)-np.sum(df.loc[:,lbl]))/np.sum(df.loc[:,lbl]) for lbl in bios_dataset.labels]
            print("class weights: ")
            print(bios_dataset.labels)
            print(class_weights)
                
            pipeline = DebiasPipeline(params, head, debias=params['debias'], validation_score=exp_config['validation_score'], class_weights=class_weights)
            
            # get samples
            train_ids = [sample['id'] for sample in bios_dataset.train_data]
            
            clf_debias_methods = ['no', 'weights', 'resample', 'add_cf', 'neutral', 'weights+neutral', 'resample+neutral']
            if params['clf_debias'] not in clf_debias_methods:
                print("clf debias method unknown. select one of these: ")
                print(clf_debias_methods)
                return
            if params['clf_debias'] == 'add_cf':
                emb = np.asarray([target_emb_all[i] for i in train_ids]+[cf_emb_all[i] for i in train_ids])
                groups = [sample['group'] for sample in bios_dataset.train_data]+[groups_cf[i] for i in train_ids]
                y = np.asarray([sample['label'] for sample in bios_dataset.train_data]+[label_cf[i] for i in train_ids])
            else:
                if params['clf_debias'] in ['no', 'weights', 'resample', 'resample_noise']:
                    emb = np.asarray([target_emb_all[i] for i in train_ids])
                elif params['clf_debias'] in ['neutral', 'weights+neutral', 'resample+neutral']:
                    emb = np.asarray([neutral_emb_all[i] for i in train_ids])
                y = np.asarray([sample['label'] for sample in bios_dataset.train_data])
                groups = [sample['group'] for sample in bios_dataset.train_data]
            
            # get sample weights
            sample_weights = []
            for sample in bios_dataset.train_data:
                cur_labels = [bios_dataset.labels[i] for i in range(len(sample['label'])) if sample['label'][i] == 1]
                group = bios_dataset.sel_groups[sample['group']]
                weights = [class_gender_weights[group][lbl]*100 for lbl in cur_labels]
                sample_weights.append(np.max(weights))
            
            if 'resample' in params['clf_debias']:
                emb, y, groups = resample(emb, y, groups, add_noise=('noise' in params['clf_debias']))
                
            # fit the whole pipeline
            if params['clf_debias'] == 'weights':
                recall, precision, f1, class_recall = pipeline.fit(emb, y, epochs=params['epochs'], optimize_theta=True, group_label=groups, weights=sample_weights)
            else:
                recall, precision, f1, class_recall = pipeline.fit(emb, y, epochs=params['epochs'], optimize_theta=True, group_label=groups)
            
            cur_result['recall'] = recall
            cur_result['precision'] = precision
            cur_result['f1'] = f1
            cur_result['class_recall'] = class_recall
            
            # compute the bias
            print("compute extrinsic biases...")
            eval_ids = [sample['id'] for sample in bios_dataset.eval_data]
            emb_eval = np.asarray([target_emb_all[i] for i in eval_ids])
            emb_eval_cf = np.asarray([cf_emb_all[i] for i in eval_ids])
            emb_eval_neutral = np.asarray([neutral_emb_all[i] for i in eval_ids])
            bios_dataset.individual_bias(pipeline.predict, emb_eval, emb_eval_cf, save_dir+'pred_cf.pickle')
            bios_dataset.group_bias(pipeline.predict, emb_eval, save_dir+'pred_raw.pickle')
            cur_result['extrinsic_individual'].append(bios_dataset.individual_biases)
            cur_result['extrinsic_classwise'].append(bios_dataset.bias_score) # class-wise GAPs
            cur_result['extrinsic'].append(np.mean(np.abs(bios_dataset.bias_score)))
            cur_result['subgroup_AUC'].append(bios_dataset.subgroup_auc)
            cur_result['BPSN'].append(bios_dataset.bpsn)
            cur_result['BNSP'].append(bios_dataset.bnsp)
            bios_dataset.group_bias(pipeline.predict, emb_eval_neutral, save_dir+'pred_neutral.pickle')
            cur_result['extrinsic_classwise_neutral'].append(bios_dataset.bias_score) # class-wise GAPs
            cur_result['subgroup_AUC_neutral'].append(bios_dataset.subgroup_auc)
            cur_result['BPSN_neutral'].append(bios_dataset.bpsn)
            cur_result['BNSP_neutral'].append(bios_dataset.bnsp)
            
            # also compute cosine scores on test data
            print("compute cosine scores on eval data...")
            if params['debias']:
                emb_eval = pipeline.debiaser.predict(np.asarray(emb_eval), pipeline.debias_k)
                emb_eval_cf = pipeline.debiaser.predict(np.asarray(emb_eval_cf), pipeline.debias_k)
                emb_eval_neutral = pipeline.debiaser.predict(np.asarray(emb_eval_neutral), pipeline.debias_k)
            target_label = [labels_cf[i] for i in eval_ids]
            target_groups = [groups_cf[i] for i in eval_ids]
            target_emb_per_group = []
            target_emb_cf_per_group = []
            target_emb_neutral_per_group = []
            for group in range(max(groups_cf)+1):
                group_name = bios_dataset.sel_groups[group]
                emb = []
                emb_cf = []
                emb_n = []
                for i in range(len(eval_ids)):
                    for lbl in classes_by_majority_group[group_name]:
                        lbl_idx = titles.index(lbl)
                        if target_label[i][lbl_idx] == 1:
                            emb.append(emb_eval[i])
                            emb_cf.append(emb_eval_cf[i])
                            emb_n.append(emb_eval_neutral[i])
                target_emb_per_group.append(emb)
                target_emb_cf_per_group.append(emb)
                target_emb_neutral_per_group.append(emb)

            # compute cosine scores
            for score in params['bias_scores']:
                if score == 'WEAT' and n_groups > 2:
                    cur_result[score].append(math.nan)
                    continue

                if score == 'DirectBias':
                    cur_score = cosine_scores[score](k=n_groups-1) # have the same dimension of bias space as SAME
                else:
                    cur_score = cosine_scores[score]()
                cur_score.define_bias_space(np.asarray(attr_emb))

                if not score == 'gWEAT':
                    if score == 'SAME' and n_groups == 2:
                        print("use signed SAME score for binary bias eval")
                        individual_biases = [cur_score.signed_individual_bias(emb_eval_cf[i]) - cur_score.signed_individual_bias(emb_eval[i]) for i in range(len(target_label))]
                        cur_result[score+'_individual'].append(individual_biases)
                        class_biases = [np.mean([cur_score.signed_individual_bias(emb_eval[i]) for i in range(len(target_label)) if target_label[i][lbl] == 1]) for lbl in range(len(target_label[0]))]
                        cur_result[score+'_classwise'].append(class_biases)
                        class_biases_n = [np.mean([cur_score.signed_individual_bias(emb_eval_neutral[i]) for i in range(len(target_label)) if target_label[i][lbl] == 1]) for lbl in range(len(target_label[0]))]
                        cur_result[score+'_classwise_neutral'].append(class_biases_n)
                    else:
                        individual_biases = [cur_score.individual_bias(emb_eval_cf[i]) - cur_score.individual_bias(emb_eval[i]) for i in range(len(target_label))]
                        cur_result[score+'_individual'].append(individual_biases)
                        class_biases = [np.mean([cur_score.individual_bias(emb_eval[i]) for i in range(len(target_label)) if target_label[i][lbl] == 1]) for lbl in range(len(target_label[0]))]
                        cur_result[score+'_classwise'].append(class_biases)
                        class_biases_n = [np.mean([cur_score.individual_bias(emb_eval_neutral[i]) for i in range(len(target_label)) if target_label[i][lbl] == 1]) for lbl in range(len(target_label[0]))]
                        cur_result[score+'_classwise_neutral'].append(class_biases_n)

                if score in ['WEAT', 'gWEAT']:
                    bias = cur_score.group_bias(target_emb_per_group)
                    bias_cf = cur_score.group_bias(target_emb_cf_per_group)
                    bias_n = cur_score.group_bias(target_emb_neutral_per_group)
                else:
                    # SAME, DirectBias, MAC
                    bias = cur_score.mean_individual_bias(emb_eval)
                    bias_cf = cur_score.mean_individual_bias(emb_eval_cf)
                    bias_n = cur_score.mean_individual_bias(emb_eval_neutral)

                cur_result[score].append(bias)
                cur_result[score+'_cf'].append(bias_cf)
                cur_result[score+'_neutral'].append(bias_n)
            
        # remove model from GPU
        if is_hugginface_model:
            lm.model.to('cpu')
        del lm
        torch.cuda.empty_cache()
        
        results.append(cur_result)
        #results_test.append(cur_result_test)
        with open(save_file, 'wb') as handle:
            pickle.dump({'params': exp_parameters, 'results': results}, handle)#, 'results_eval': results_test}, handle)
        print()
        
    with open(save_file, 'wb') as handle:
        pickle.dump({'params': exp_parameters, 'results': results}, handle)#, 'results_eval': results_test}, handle)
    print('done')


def main(argv):
    config_path = ''
    min_iter = 0
    max_iter = -1
    try:
        opts, args = getopt.getopt(argv, "hc:", ["config="])
    except getopt.GetoptError:
        print('bios_experiment.py -c <config>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('bios_experiment.py -c <config>')
            sys.exit()
        elif opt in ("-c", "--config"):
            config_path = arg

    print('use config:' + config_path)

    with open(config_path, 'r') as f:
        exp_config = json.load(f)

    run_clf_experiments(exp_config)


if __name__ == "__main__":
    main(sys.argv[1:])
