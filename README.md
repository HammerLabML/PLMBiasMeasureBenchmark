# PLMBiasMeasureBenchmark
This repository contains the implementation and configs used for the paper [So can we use intrinsic bias measures or not?](https://www.scitepress.org/Papers/2023/116937/116937.pdf) by Sarah Schröder, Alexander Schulz, Philip Kenneweg and  Barbara Hammer.


## Requirements
All requirements are listed in ```requirements.txt```.  
The [embeddings](https://github.com/UBI-AGML-NLP/Embeddings) and [embedding-bias-eval](https://github.com/HammerLabML/EmbeddingBiasScores) packages must be installed from source.   


## Installation
Download this repository, install the above mentioned requirements.

## Reproducing our experiments

Use ```configs/icpram22.yaml``` and run ```multi_attr_bias_test.py``` to produce a batch of pretrained models with the parameters used in our experiments. Follow the steps in ```icpram_eval.ipynb``` to evaluate the bias measures and visualize the results.  
Alternatively, download our trained [models and results]() to skip the computationally expensive part.

## Running custom experiments


## Running on multiple machines
Training a larger number of language models with ```multi_attr_bias_test.py```might take a long time on one machine. You can divide the task onto multiple machines by using the ```min``` and ```max``` parameters to specify the range of model ids that should be trained and evaluated on the current machine. Afterwards simply merge the results in one directory.
