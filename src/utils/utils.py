import random
import logging
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # filename='myapp.log', 


valid_objectives = ['MLM', 'MLM_lazy', 'NSP']
valid_masking_strategies = ['random', 'attribute', 'non_attribute', 'target']
valid_eval_strategies = ['non_attribute', 'target']

def check_config(config):
    """
    Test if config contains all necessary keys and valid entries for free-text parameters.

    In case of invalid or missing options (depending on the parameter), either
    - set a default and print info
    - print an error and exit
    
    Args:
        config (dict): The config in form of a dictionary.
            
    Returns:
        None: Prints infos/ errors (and exits if necessary).
    """

    # these can be replaced by default values when missing
    if 'batch_size' not in config.keys():
        config['batch_size'] = 8
        logger.info("no batch size defined in the config, default to 8")
    if 'minP' not in config.keys():
        logger.info("minP not in config, use default values")
        config['minP'] = [0.0, 0.1, 0.2]
    if 'maxP' not in config.keys():
        logger.info("maxP not in config, use default values")
        config['maxP'] = [0.8, 0.9, 1.0]
    if 'iterations' not in config.keys():
        logger.info("iterations not in config, use default")
        config['iterations'] = 5
    if 'random_seed' not in config.keys():
        logger.info("random_seed not in config, use default")
        config['random_seed'] = 42
    if 'pretrained_model' not in config.keys():
        logger.info("pretrained model not in config, use default")
        config['pretrained_model'] = 'bert-base-uncased'
    if 'epochs' not in config.keys():
        logger.info("epochs not specified in config, use default")
        config['epochs'] = 5

    # these cannot be replaced by default values
    if 'template_file' not in config.keys():
        logger.error("template_file missing from config")
        exit(1)
    if 'results_dir' not in config.keys():
        logger.error("results_dir missing from config")
        exit(1)
    if 'masking_strategy' not in config.keys() or config['masking_strategy'] not in valid_masking_strategies:
        logger.error("Did not specify a valid masking strategy. Choose one of these: ", valid_masking_strategies)
        exit(1)
    if 'mask_prob' not in config.keys() and config['masking_strategy'] in ['non_attribute', 'random']:
        logger.error("When using 'random' or 'non_attribute' masking strategy, the 'mask_prob' parameter must be "
              "specified.")
        exit(1)
    #if 'eval_strategy' not in config.keys() or config['eval_strategy'] not in valid_eval_strategies:
    #    print("error: Did not specify a valid eval strategy. Choose one of these: ", valid_eval_strategies)
    #    exit(0)


def check_attribute_occurence(template_config: dict):
    """
    Calculates the relative occurrence rate of protected attribute placeholders in training and test templates.
    
    Args:
        template_config (dict): A dictionary containing:
            - 'protected_attr': List of strings representing attribute names (e.g., ['GENDER', 'ETHNICITY']).
            - 'templates_train': List of template strings for the training set, which contain attribute placeholder (e.g. 'GENDER*').
            - 'templates_test': List of template strings for the testing set, which contain attribute placeholder (e.g. 'GENDER*').
            
    Returns:
        None: Prints a dictionary with the occurence rates of protected attributes.
    """
    logger.info("check occurence of protected groups in the training and test templates...")
    protected_attributes = template_config['protected_attr']
    attribute_stats = {}
    for attr in protected_attributes:
        attribute_stats.update({attr: {'train': 0, 'test': 0}})

    n_train = len(template_config['templates_train'])
    n_test = len(template_config['templates_test'])

    for temp in template_config['templates_train']:
        for attr in protected_attributes:
            for key in template_config[attr]['KEYS']:
                if key in temp:
                    attribute_stats[attr]['train'] += 1

    for temp in template_config['templates_test']:
        for attr in protected_attributes:
            for key in template_config[attr]['KEYS']:
                if key in temp:
                    attribute_stats[attr]['train'] += 1

    for attr, entry in attribute_stats.items():
        entry['train'] /= n_train
        entry['test'] /= n_test

    print(attribute_stats)

