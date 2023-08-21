import os
import yaml
import getopt
import sys

from utils import create_bias_distribution, check_config, check_attribute_occurence, create_masked_dataset
from embedding import BertHuggingfaceMLM, BertHuggingface
from unmasking_bias import PLLBias


def test_templates(template_config):
    target_domain = template_config['target']
    target_words = template_config[target_domain]
    protected_attributes = template_config['protected_attr']
    versions_per_attr = {attr: len(template_config[attr]) for attr in protected_attributes}

    # test training and tes templates
    for template in template_config['templates_train']+template_config['templates_test']:
        sent = template

        assert target_domain in template, "did not find target placeholder in training template: "+template
        found_attr = False
        for attr in protected_attributes:
            found_attr = found_attr or attr in template

            if attr in template:
                for i in range(versions_per_attr[attr]):
                    cur_attr = attr+str(i)
                    sent = sent.replace(cur_attr, template_config[attr][i][0])
                assert (attr not in sent), "after replacing all attribute versions, the placeholder (" + attr\
                                               + ")still remains:\n" + sent
        assert found_attr, "did not find any attribute placeholder in training template: "+template


def init(config):

    print("load templates and protected attributes...")
    with open(config['template_file'], 'r') as f:
        template_config = yaml.safe_load(f)

    print("create the directory for experiment results...")
    if not os.path.isdir(config['results_dir']):
        os.makedirs(config['results_dir'])
    log_config = config['results_dir']+'/config.yaml'
    print(config)

    with open(log_config, 'w') as file:
        yaml.dump(config, file)

    test_templates(template_config)

    # download model if necessary
    if 'MLM' in config['objective']:  # MLM or MLM_lazy
        lm = BertHuggingfaceMLM(model_name=config['pretrained_model'], batch_size=config['batch_size'])
        # test unmasking/ ground truth bias score
        # TODO: update these
        pll = PLLBias(lm.model, lm.tokenizer, batch_size=config['batch_size'])

        sent1 = ['how are u', 'hello u', 'hey u']
        sent2 = ['how are you', 'hello you', 'hey you']
        pll.compare_sentence_likelihood(sent1, sent2)

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
