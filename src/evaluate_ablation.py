import os
import getopt
import sys
import pickle
import yaml

import numpy as np
import math
import pandas as pd
import scipy

import ast
import re
import random
from tqdm import tqdm

import plotly.graph_objects as go
import plotly.express as px
from matplotlib.colors import to_rgba

import torch
from utils import (create_bias_distribution, check_config, check_attribute_occurence, create_masked_dataset, templates_to_train_samples, templates_to_eval_samples, 
                   evaluate_mlm, forward_mlm_for_bias_eval, load_wikitext, mask_texts)

from embedding import BertHuggingfaceMLM
from unmasking_bias import PLLBias


PALETTES = {
    "Plotly": px.colors.qualitative.Plotly,        # 10
    "D3":     px.colors.qualitative.D3,            # 10
    "Set1":  px.colors.qualitative.Set1,           # 9
    "Dark2": px.colors.qualitative.Dark2,          # 8
    "Vivid": px.colors.qualitative.Vivid,          # 10
    "Alphabet": px.colors.qualitative.Alphabet,    # 26
    "Light24": px.colors.qualitative.Light24,      # 24
    "Dark24": px.colors.qualitative.Dark24,        # 24
}

def pick_colors(n: int) -> list[str]:
    """Return n qualitative colors, extending the palette by cycling if needed."""
    # choose the smallest palette that covers n; fall back to the largest
    candidates = sorted(
        (colors for colors in PALETTES.values() if len(colors) >= n),
        key=len,
    )
    palette = candidates[0] if candidates else max(PALETTES.values(), key=len)

    # extend by cycling through the palette if n > len(palette)
    if len(palette) < n:
        palette = [palette[i % len(palette)] for i in range(n)]
    return palette[:n]


SEED_KEY = 'random_seed'

# TODO: get a dir with result subdirs, scan each for config and results, 
#       get parameters+random seeds form config (if not in result),
#       concat results (identified by param)
#       plot results (single metric, line per param choice, deviation over random seed, x axis: epochs)
#       for wiki ratio and masking ablation separately

def color_map(df: pd.DataFrame, column: str) -> dict:
    """Map each unique value in df[col] to a distinct color."""
    uniques = pd.unique(df['c'])
    return dict(zip(uniques, pick_colors(len(uniques))))

def create_performance_plot(xvalues: list[float],
                            yvalues: list[float], 
                            errors: list[float],
                            colors: list[str],
                            title: str = 'dummy title', 
                            legend_title: str = 'param', 
                            filename: str = None,
                            width=1000, height=600):
    """
    Creates a Plotly line plot with error bars, colored.
    """
    df = pd.DataFrame({'x': xvalues, 'y': yvalues, 'c': colors, 'err': errors})
    cmap = color_map(df, column='c')
    fig = go.Figure()
    
    for c, grp in df.groupby('c'):
        # errors
        upper_bound = [m + s for m, s in zip(grp['y'], grp['err'])]
        lower_bound = [m - s for m, s in zip(grp['y'], grp['err'])]

        r, g, b = cmap[c][4:-1].split(',')
        color_rgba_str = f"rgba({r}, {g}, {b}, {0.2})"
        
        x_hull = grp['x'].tolist() + grp['x'].tolist()[::-1]
        y_hull = upper_bound + lower_bound[::-1]

        fig.add_trace(go.Scatter(
            x=x_hull,
            y=y_hull,
            fill='toself',
            fillcolor=color_rgba_str,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))

        # mean
        fig.add_trace(go.Scatter(
            x=grp['x'], y=grp['y'],
            mode='lines+markers',
            name=str(c),
            line=dict(color=cmap[c], width=2, dash='dash')
        ))

    # layout
    font_sizes = {
        "title": 28,
        "axis": 24,
        "legend": 24,
        "tick": 18
    }
    layout_updates = {
        "xaxis": {
            "title": {
                "text": "Epoch", 
                "font": {"size": font_sizes["axis"]}
                }
        },
        "yaxis": {
            "title": {
                "text": title, 
                "font": {"size": font_sizes["axis"]}
                }
        },
        "hovermode": "x unified",
        "template": "plotly_white",
        "width": width,
        "height": height,
        "legend": {
            "title": {"text": legend_title, "font": {"size": font_sizes["legend"], "family": "Arial, sans-serif"}},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": font_sizes["legend"], "family": "Arial, sans-serif"}
        }
    }
    fig.update_layout(**layout_updates)

    fig.show()

    if filename is not None:
        try:    
            fig.write_image(f"{filename}.png")
            print(f"Saved plot: {filename}.png")
        except Exception as e:
            print(f"Error saving plot as png: {e}")
            fig.write_html(f"{filename}.html", auto_open=False)
            print("saved as html instead")


def str_to_list_float(s):
    if isinstance(s, str):
        try:
            lst = ast.literal_eval(s)
            return [float(x) for x in lst]
        except (ValueError, SyntaxError):
            print(f"Warning: Could not parse '{s}'")
            s_ = s.replace('[','').replace(']','')
            return [float(x) for x in s_.split(' ') if x != '']
#            return []
    return s


def create_ablation_plot(result_path: str, param_key: str):
    dfs = []

    for entry in os.listdir(result_path):
        subdir_path = os.path.join(result_path, entry)
        if not os.path.isdir(subdir_path):
            continue

        # load config
        yaml_path = os.path.join(subdir_path, "config.yaml")
        try:
            with open(yaml_path, "r") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: no config.yaml in {subdir_path}")
            continue

        csv_path = os.path.join(subdir_path, "results.csv")
        df = pd.read_csv(csv_path)
        
        # add subdir path, ablation param value and random seed
        df["subdir"] = entry
        df[param_key] = config[param_key]
        df[SEED_KEY] = config[SEED_KEY]

        dfs.append(df)

    results = pd.concat(dfs, ignore_index=True)
    results.sort_values(by='subdir').to_csv('test.csv', index=False)

    for score, title in [('r_test', 'Pearson Correlation R ↑'), ('acc', 'Accuracy ↑'), ('ppl', 'Perplexity ↓')]:
        results[score] = results[score].apply(str_to_list_float)
        scores = []
        errors = []
        epochs = []
        params = []
        for pval in set(results[param_key]):
            sel = results[results[param_key] == pval]
            cur_scores = np.vstack(sel[score])
            mean_scores = np.mean(cur_scores, axis=0)
            scores += mean_scores.tolist()
            errors += np.std(cur_scores, axis=0).tolist()
            epochs += list(range(len(mean_scores)))
            params += [pval]*len(mean_scores)

        create_performance_plot(xvalues=epochs, yvalues=scores, errors=errors, colors=params, 
                                title=title, legend_title=param_key, filename=result_path+'/'+score)
        



def main(argv):
    result_path = ''
    param_key = ''
    try:
        opts, args = getopt.getopt(argv, "hr:p:", ["results=", "parameter="])
    except getopt.GetoptError:
        print('evaluate_ablation.py -r <result dir> -p <parameter key>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('evaluate_ablation.py -r <result dir> -p <parameter key>')
            sys.exit()
        elif opt in ("-r", "--results"):
            result_path = arg
        elif opt in ("-p", "--param"):
            param_key = arg

    print('config is ' + result_path)
    print('parameter key is ' + param_key)

    assert os.path.isdir(result_path), "expected an existing directory with results"

    create_ablation_plot(result_path, param_key)


if __name__ == "__main__":
    main(sys.argv[1:])
