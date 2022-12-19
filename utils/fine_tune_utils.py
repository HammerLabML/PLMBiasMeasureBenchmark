import random
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def create_bias_distribution(n_groups, target_words, minP=0.0, maxP=1.0):
    assert minP < 1.0 / n_groups, "minP must be in [0, 1/n_groups)"

    probs_by_target = {}
    for target in target_words:
        probs_by_target.update({target: {}})

    for target in target_words:
        P = 1.0
        groups = list(range(n_groups))
        while len(groups) > 1:
            i = random.choice(groups)
            groups.remove(i)
            p = random.uniform(minP, min(maxP, P - minP * len(groups)))
            P -= p
            probs_by_target[target].update({i: p})

        # last group
        i = groups[0]
        p = P
        probs_by_target[target].update({i: p})

    return probs_by_target
