#!/bin/bash

# Scripts for replicating the results reported in our EMNLP '19 paper: https://www.aclweb.org/anthology/D19-1468.pdf
# Before running this script, make sure you've installed dependencies and downloaded the required data (see README)
mkdir ../experiments/

## Restaurant Reviews in SemEval dataset (Table 4)
logdir="../experiments/Restaurants"
mkdir $logdir

# Student-W2V
./run_5_times.sh semeval --lr 0.01 --no_seed_weights --disable_gpu --logdir $logdir/$(date +'%h%d_%H-%M-%S-%N')_W2V --batch_size 64 --emb_dropout 0.2 --loss SmoothCrossEntropy

# Student-BERT
./run_5_times.sh semeval --lr 0.01 --no_seed_weights --disable_gpu --logdir $logdir/$(date +'%h%d_%H-%M-%S-%N')_BERT --batch_size 64 --emb_dropout 0.2 --loss SmoothCrossEntropy --use_bert


## Amazon Product Reviews in OPOSUM dataset (Table 3)
# To run OPOSUM scripts make sure you first download the OPOSUM dataset (see data/download_data.sh) and then uncomment the following lines
#  logdir="../experiments/Products"
#  mkdir $logdir

# Student-W2V
#  ./run_5_times.sh pairs --lr 0.0005 --no_seed_weights --logdir $logdir/$(date +'%h%d_%H-%M-%S-%N')_W2V --emb_dropout 0.5 --weight_decay 0.1 --loss SmoothCrossEntropy

# Student-BERT
#  ./run_5_times.sh pairs --lr 0.00005 --no_seed_weights --logdir $logdir/$(date +'%h%d_%H-%M-%S-%N')_BERT --emb_dropout 0.5 --weight_decay 0.0001 --use_bert
