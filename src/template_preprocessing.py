import pandas as pd
import yaml
import numpy as np
import sys
import getopt


def custom_split(df, samples_train_only, samples_test, percent_test=0.4):
    """
    Splits data into Train, Validation, and Test sets following specific rules:
    - Test and validation samples are only taken from 'samples_test'
    - The test set size is specified via 'percent_test' and the training set is the rest
    - The validation set is the training set excluding samples from 'samples_train_only' - intended to distinguish generalization and overfitting
    """
    
    # determine total rows and split sizes
    total_rows = len(samples_train_only) + len(samples_test)
    
    if total_rows == 0:
        raise ValueError("No data available in either dataframe.")

    assert (0.0 < percent_test < 1.0)
    #assert (0.0 < percent_val < 1.0) 
    #n_val = int(np.round(total_rows * percent_val))
    n_test = int(np.round(total_rows * percent_test))
    n_train = len(df) - n_test # - n_val
    
    # check sufficent test/val samples
    percent_train = 1.0 - percent_test #- percent_val
    #assert (percent_train < 1.0 and percent_train > 0)
    if len(samples_train_only) > n_train:
        print("warning: training set will be larger than requested (%.2f) (too many samples not eligible for test/val split)" % (percent_train))

    # shuffle
    shuffled_test = samples_test.sample(frac=1, random_state=42).reset_index(drop=True)

    # extract test and validation split first
    df_test = shuffled_test.iloc[:n_test]
    #df_val = shuffled_test.iloc[n_test : n_test + n_val]
    
    # train set is the remaining samples from 'samples_test' plus 'samples_train_only', shuffle again
    #df_test_remainder = shuffled_test.iloc[n_test + n_val :]
    df_test_remainder = shuffled_test.iloc[n_test :]
    df_val = df_test_remainder
    df_train = pd.concat([samples_train_only, df_test_remainder], ignore_index=True)
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_train, df_val, df_test


def create_templates(template_collection_file: str, template_config_file: str):
    """
    Creates an experiment-ready template config file from a .csv (basically list of templates) and the base yaml file.
    The yaml file already specifies the targets and attributes. Template sentences will be tested for the occurence of a target placeholder and at least one for attributes.
    The .csv includes a column 'template' with template sentences (and optionally auxiliary columns that specify style or present attributes). All templates should contain
    placeholders for the targets and one of the attributes specified in the yaml.

    The templates will be split into train and test set (depending on number of attributes) and another validations set will be created that contains all
    samples from the test set that work with the evaluation scheme (just one attribute). The validation set is expected to be used to estimate the perforance on the train set!


    Args:
        template_collection_file (str): The filename of the .csv with all templates.
        template_config_file (str): The filename of the template config (yaml file).

    Returns:
        None: Saves the resulting template config file.
    """

    # load yaml with attr/ target keys and csv with templates
    with open(template_config_file, 'r') as f:
        template_config = yaml.safe_load(f)

    df = pd.read_csv(template_collection_file, sep=';')

    # filter required styles
    if 'styles' in template_config.keys():
        print("only use templates of the following styles: ", template_config['styles'])
        print(len(df))
        df = df[df['style'].isin(template_config['styles'])].reset_index(drop=True)
        print(len(df))

    # get templates, attribute and target keys
    templates = list(df['template'])
    attributes = template_config['protected_attr']
    keys_by_attr = {attr: template_config[attr]['KEYS'] for attr in attributes}
    target = template_config['target']

    # assert every template has a target key
    for template in templates:
        assert target in template, "found template without target key: "+template

    count_per_key = {key: 0 for attr in attributes for key in keys_by_attr[attr]}
    for i, template in enumerate(templates):
        for attr in attributes:
            df.loc[i, attr] = 0
            for key in keys_by_attr[attr]:
                df.loc[i, attr] += template.count(key)
                count_per_key[key] += template.count(key)
    print("occurences per key:")
    print(count_per_key)

    df.to_csv(template_collection_file.replace('.csv', '_.csv'), index=False, sep=';')

    #print(df)
    condition_test = df[attributes] <= 1
    condition_not_test = df[attributes] > 1
    samples_train_only = df[condition_not_test.any(axis=1)]
    samples_test_valid = df[condition_test.all(axis=1)]

    condition_zero = df[attributes] == 0
    samples_zero = df[condition_zero.all(axis=1)]
    print("samples without any attributes:")
    print(samples_zero)

    print("got %i samples in total" % len(df))
    print("found %i samples with more than one key per attribute (not eligible for test/val split)" % len(samples_train_only))

    # split data
    df_train, df_val, df_test = custom_split(df, samples_train_only, samples_test_valid, percent_test=0.4)


    for attr in attributes:
        print("for %s got %i / %i / %i (train/val/test) samples" % (attr, len(df_train[df_train[attr] >= 1]), len(df_val[df_val[attr] >= 1]), len(df_test[df_test[attr] >= 1])))


    template_config['templates_train'] = list(df_train['template'])
    template_config['templates_val'] = list(df_val['template'])
    template_config['templates_test'] = list(df_test['template'])

    with open(template_config_file, 'w') as f:
        yaml.dump(template_config, f, default_flow_style=False, sort_keys=False)




def main(argv):
    csv_path = ''
    yaml_path = ''
    try:
        opts, args = getopt.getopt(argv, "hc:", ["csv=", "yaml="])
    except getopt.GetoptError:
        print('template_preprocessing.py --csv <template collection> --yaml <template config>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('template_preprocessing.py --csv <template collection> --yaml <template config>')
            sys.exit()
        elif opt in ("--csv"):
            csv_path = arg
        elif opt == "--yaml":
            yaml_path = arg

    assert '.csv' in csv_path, "expected a csv file"
    assert '.yaml' in yaml_path, "expected a yaml file"

    print(csv_path)
    print(yaml_path)

    create_templates(template_collection_file=csv_path, template_config_file=yaml_path)


if __name__ == "__main__":
    main(sys.argv[1:])
