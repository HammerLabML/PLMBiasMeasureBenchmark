import os
import yaml
import getopt
import sys

from utils import create_bias_distribution, check_config, check_attribute_occurence, create_masked_dataset
from embedding import BertHuggingfaceMLM, BertHuggingface
from unmasking_bias import PLLBias


def init(config):

    print("load templates and protected attributes...")
    with open(config['template_file'], 'r') as f:
        tmp = yaml.safe_load(f)

    print("create the directory for experiment results...")
    if not os.path.isdir(config['results_dir']):
        os.makedirs(config['results_dir'])
    log_config = config['results_dir']+'/config.yaml'
    print(config)

    with open(log_config, 'w') as file:
        yaml.dump(config, file)

    #  TODO: check templates (that each one contains at least one attribute and a target placeholder (typos!)

    # download model if necessary
    if 'MLM' in config['objective']:  # MLM or MLM_lazy
        lm = BertHuggingfaceMLM(model_name=config['pretrained_model'], batch_size=config['batch_size'])
        # test unmasking/ ground truth bias score
        # TODO: update these
        pll = PLLBias(lm.model, lm.tokenizer)
        pll.PLL_compare_sent("hello you", "hello u")

        lm.embed(["test"])
    else:
        print("other training objective not supported for testing")


def main(argv):
    config_path = ''
    try:
        opts, args = getopt.getopt(argv, "hc:", ["config=", "min=", "max="])
    except getopt.GetoptError:
        print('multi_attr_bias_test.py -c <config>')
        sys.exit(1)
    for opt, arg in opts:
        if opt == '-h':
            print('multi_attr_bias_test.py -c <config>')
            sys.exit()
        elif opt in ("-c", "--config"):
            config_path = arg

    print('config is ' + config_path)

    with open(config_path, 'rb') as f:
        config = yaml.safe_load(f)
        check_config(config)
    print(config)

    init(config)


if __name__ == "__main__":
    main(sys.argv[1:])
