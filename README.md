# PLMBiasMeasureBenchmark
This repository contains the implementation and configs used for the paper "The SAME score: Improved cosine based measure for semantic bias" by Sarah Schröder, Alexander Schulz and  Barbara Hammer (accepted at IJCNN 2024).

## Requirements
All requirements are listed in ```requirements.txt```.  
The [embeddings](https://github.com/UBI-AGML-NLP/Embeddings) and [embedding-bias-eval](https://github.com/HammerLabML/EmbeddingBiasScores) packages must be installed from source.   

## Installation
Download this repository, install the above mentioned requirements.

## Reproducing our experiments

The code for our experiments was splitted into python scripts that cover the computationally expensive part and jupyter notebooks for the final evaluation and displaying results. In the following we give instructions on how to reproduce the experiments on the respective datasets. Configs could be adapted to test other sets of models or hyperparameters.

### BIOS
Follow the instructions in [this repo](https://github.com/HammerLabML/MeasuringFairnessWithBiasedData) to obtain the supervised subset of the BIOS dataset we used in the paper. In principle, one could run the experiment with the entire [BIOS dataset](https://github.com/microsoft/biosbias), but that might require some adaptions to the code and the configs. Adapt the field "bios_file" in the config ```configs/bios_exp.json``` to match the location of the BIOS dataset. Run
> python3 bios_experiment.py -c configs/bios_exp.json  
to train the models and compute bias scores. Then run ```bios.ipynb``` for the evaluation.

### Jigsaw Toxicity
The Jigsaw Unintended Bias dataset is available as a [Huggingface model](https://huggingface.co/datasets/google/jigsaw_unintended_bias). However, the data has to be downloaded manually from the Kaggle competition and the path specified in the config file. Run
> python3 jigsaw_experiment.py -c configs/jigsaw_<gender|race|religion>.json  
to train the models and compute bias scores. Then run ```jigsaw.ipynb``` for the evaluation.

### CrowSPairs
Run
> python3 mlm_experiment.py -c configs/mlm_exp.json  
to train the models and compute bias scores. Then run ```crowspairs.ipynb``` for the evaluation.
