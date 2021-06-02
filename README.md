# ISWD
Iterative Seed Word Distillation (EMNLP 2019)

This repo holds the code for our weakly-supervised co-training framework, ISWD, described in our EMNLP 2019 paper: "[Leveraging Just a Few Keywords for Fine-Grained Aspect Detection Through Weakly Supervised Co-Training
](https://www.aclweb.org/anthology/D19-1468/)" 

## Overview of ISWD
ISWD is a teacher-student framework for training deep neural networks for fine-grained aspect detection using user-defined seed words, i.e., indicative keywords. ISWD can be used for tasks where it is expensive to manually collect large-scale labeled training data.

ISWD leverages just a few **keywords** and a large amount of **unlabeled data** through a **teacher-student** architecture:
* **Teacher**: a simple bag-of-seed-words classifier that uses seed words to predict aspect probabilities
* **Student**: an embedding-based neural network (can have any architecture) that uses both seed words and non-seed words (context) to predict aspect probabilities. The student predicts aspects even if no seed words appear in the text

The student is trained through the **distillation** objective that is to minimize the (soft) cross-entropy between the student's predictions and the teacher's predicted probabilities. 

Our [EMNLP'19 paper](https://www.aclweb.org/anthology/D19-1468/) describes our ISWD framework and experimental results in detail. A preliminary version of our work was presented at our [LLD@ICLR'19 paper](https://openreview.net/pdf?id=HyxoAoxOwN).

## Installation

First, create a conda environment running Python 2.7: 
```
conda create --name iswd python=2.7
conda activate iswd
```

Then, install the required dependencies:
```
pip install -r requirements.txt
```

## Download Data
For reproducibility, you can directly download our pre-processed data files: 

```
cd data
bash download_data.sh
```

The original product review datasets (OPOSUM) are available [here](https://github.com/stangelid/oposum). The original restaurant review datasets (SemEval) are available [here](https://alt.qcri.org/semeval2016/task5/). If you are using those datasets, please cite the corresponding papers. 


## Running ISWD
To replicate our EMNLP '19 experiments, you can directly run our bash script:
```
cd scripts
bash run_experiments.sh
```
The above script will run ISWD and report results under a new "experiments" folder. 


## Citations

```
@inproceedings{karamanolakis2019leveraging,
  title={Leveraging Just a Few Keywords for Fine-Grained Aspect Detection Through Weakly Supervised Co-Training},
  author={Karamanolakis, Giannis and Hsu, Daniel and Gravano, Luis},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={4603--4613},
  year={2019}
}

@inproceedings{karamanolakis2019seedwords,
  title={Training Neural Networks for Aspect Extraction Using Descriptive Keywords Only},
  author={Karamanolakis, Giannis and Hsu, Daniel and Gravano, Luis},
booktitle={Proceedings of the Second Learning from Limited Labeled Data Workshop},
  year={2019}
}
```







