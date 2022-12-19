
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
    if 'add_wiki_sent' not in config.keys():
        print("add_wiki_sent not speicifed in config, use default")
        config['add_wiki_sent'] = False
    # these cannot be replaced by default values
    if 'template_file' not in config.keys():
        print("error: template_file missing from config")
        exit(0)
    valid_objectives = ['MLM', 'MLM_lazy', 'NSP']
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
