import random

valid_objectives = ['MLM', 'MLM_lazy', 'NSP']
valid_masking_strategies = ['random', 'attribute', 'non_attribute', 'target']


def check_config(config):
    # these can be replaced by default values when missing
    if 'batch_size' not in config.keys():
        config['batch_size'] = 8
        print("no batch size defined in the config, default to 8")
    if 'minP' not in config.keys():
        print("minP not in config, use default values")
        config['minP'] = [0.0, 0.1, 0.2]
    if 'maxP' not in config.keys():
        print("maxP not in config, use default values")
        config['maxP'] = [0.8, 0.9, 1.0]
    if 'iterations' not in config.keys():
        print("iterations not in config, use default")
        config['iterations'] = 5
    if 'random_seed' not in config.keys():
        print("random_seed not in config, use default")
        config['random_seed'] = 42
    if 'pretrained_model' not in config.keys():
        print("pretrained model not in config, use default")
        config['pretrained_model'] = 'bert-base-uncased'
    if 'epochs' not in config.keys():
        print("epochs not specified in config, use default")
        config['epochs'] = 5
    # these cannot be replaced by default values
    if 'template_file' not in config.keys():
        print("error: template_file missing from config")
        exit(0)
    if 'objective' not in config.keys():
        print("error: objective missing from config")
        exit(0)
    else:
        if config['objective'] not in valid_objectives:
            print("error: objective must be one of the following: ", valid_objectives)
    if 'results_dir' not in config.keys():
        print("error: results_dir missing from config")
        exit(0)
    if 'target_words' not in config.keys():
        print("error: target_words missing from config")
        exit(0)
    if 'masking_strategy' not in config.keys() or config['masking_strategy'] not in valid_masking_strategies:
        print("error: Did not specify a valid masking strategy. Choose one of these: ", valid_masking_strategies)
        exit(0)
    if 'mask_prob' not in config.keys() and (config['masking_strategy'] in ['non_attribute', 'random']):
        print("error: When using 'random' or 'non_attribute' masking strategy, the 'mask_prob' parameter must be "
              "specified.")
        exit(0)


def check_attribute_occurence(template_config: dict):
    print("check occurence of protected groups in the training and test templates...")
    protected_attributes = template_config['protected_attr']
    attribute_stats = {}
    for attr in protected_attributes:
        attribute_stats.update({attr: {'train': 0, 'test': 0}})

    n_train = len(template_config['templates_train'])
    n_test = len(template_config['templates_test'])

    for temp in template_config['templates_train']:
        for attr in protected_attributes:
            if attr in temp:
                attribute_stats[attr]['train'] += 1

    for temp in template_config['templates_test']:
        for attr in protected_attributes:
            if attr in temp:
                attribute_stats[attr]['test'] += 1

    for attr, entry in attribute_stats.items():
        entry['train'] /= n_train
        entry['test'] /= n_test

    print(attribute_stats)

