# PLMBiasMeasureBenchmark
This repository contains the implementation and configs used for experiments of the following papers:
- "So can we use intrinsic bias measures or not?", Sarah Schröder, Alexander Schulz, Philip Kenneweg and  Barbara Hammer, published at ICPRAM (2022) [Conference Paper](https://www.scitepress.org/Papers/2023/116937/116937.pdf)
- "Semantic Properties of Cosine Based Bias Scores for Word Embeddings", Sarah Schröder, Alexander Schulz, Fabian Hinder and Barbara Hammer published at ICPRAM (2023) [Conference Paper](https://www.scitepress.org/PublishedPapers/2024/125772/) [ArXiv Preprint](https://arxiv.org/abs/2401.15499)
- "The SAME score: Improved cosine based measure for semantic bias", Sarah Schröder, Alexander Schulz and Barbara Hammer, published at IJCNN (2024) [Conference Paper](https://ieeexplore.ieee.org/abstract/document/10651275) [ArXiv Preprint](https://arxiv.org/abs/2203.14603)


## Requirements
All requirements are listed in ```requirements.txt```.  
The [embeddings](https://github.com/UBI-AGML-NLP/Embeddings) and [embedding-bias-eval](https://github.com/HammerLabML/EmbeddingBiasScores) packages must be installed from source.   


## Installation
Download this repository, install the above mentioned requirements.

## Reproducing our ICPRAM 2022 Paper

Use ```configs/icpram22.yaml``` and run ```src/experiments/icpram22/multi_attr_bias_test.py``` to produce a batch of pretrained models with the parameters used in our experiments. Follow the steps in ```icpram_eval.ipynb``` to evaluate the bias measures and visualize the results.  
Alternatively, download our trained [models and results]() to skip the computationally expensive part.

### Running on multiple machines
Training a larger number of language models with ```src/experiments/icpram22/multi_attr_bias_test.py``` might take a long time on one machine. You can divide the task onto multiple machines by using the ```min``` and ```max``` parameters to specify the range of model ids that should be trained and evaluated on the current machine. Afterwards simply merge the results in one directory.


## Reproducing our ICPRAM 2023 Paper

Run the notebook ```src/experiments/icpram23/db_weat_examples.ipynb```.


## Reproducing our IJCNN 2024 Paper

TODO


## Other experiments

TODO