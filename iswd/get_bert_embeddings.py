#!/usr/bin/env python
import os
from os.path import expanduser
home = expanduser("~")
import sys
import h5py
import argparse
from datetime import datetime
from time import time

import numpy as np
from numpy.random import permutation, seed
from scipy.cluster.vq import kmeans
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_uniform
from torch.nn.utils import clip_grad_norm

from tensorboardX import SummaryWriter

from sklearn.externals import joblib


import BaselineDataHandler
from BaselineDataHandler import DataHandler

sys.path.append(os.path.join(home, "code/research_code/Spring_2018/TextModules/"))
from Evaluator import Evaluator
from Logger import get_logger
from model_library import AspectCLF, StudentBoWCLF


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--domain', help="Domain name (without extension)", type=str, default='pairs')

    # Trainer Specific
    parser.add_argument('--logdir', help="log directory for tensorboard", type=str, default='/home/gkaraman/data1/experiments/absa_joint')
    parser.add_argument('--debug', help="Enable debug mode", action='store_true')
    parser.add_argument('--num_epochs', help="Number of epochs (default: 25)", type=int, default=25)
    parser.add_argument('--loss', help="Loss Function (CrossEntropy / NLL)", type=str, default='CrossEntropy')
    parser.add_argument('--optimizer', help="Optimizer (Adam / Adadelta)", type=str, default='Adam')
    parser.add_argument('--lr', help="Learning rate (default: 0.0001)", type=float, default=0.0001)
    parser.add_argument('--weight_decay', help="Weight Decay", type=float, default=0.0)
    parser.add_argument('--momentum', help="Momentum (used for optimizer=SGD)", type=float, default=0.9)
    parser.add_argument('--report_every', help="Report every x number of batches", type=int, default=50)
    parser.add_argument('--cuda_device', help="CUDA Device ID", type=int, default=0)
    parser.add_argument('--batch_size', help="Batch Size", type=int, default=1024)
    parser.add_argument('--target_metric', help="Target Metric to report", type=str, default='micro_average_f1')
    parser.add_argument('--version', help="Run # (1..5)", type=int, default=1)

    # Domain Specific
    parser.add_argument('--test_data', help="hdf5 file of test segments", type=str, default='')
    parser.add_argument('--min_len', help="Minimum number of non-stop-words in segment (default: 2)", type=int, default=2)
    parser.add_argument('--num_aspects', help="Number of aspects (default: 9)", type=int, default=9)
    parser.add_argument('--aspect_seeds', help='file that contains aspect seed words (overrides number of aspects)', type=str, default='')
    parser.add_argument('-q', '--quiet', help="No information to stdout", action='store_true')
    parser.add_argument('--num_seeds', help="Number of seed words to use (default: 30)", type=int, default=30)
    parser.add_argument('--no_seed_weights', help="Forcing the *unweighted* avg of seed word embeddings", action='store_true')
    parser.add_argument('--batch_norm', help="Batch normalization on segment encodings", action='store_true')
    parser.add_argument('--emb_dropout', help="Dropout at the segment embedding layer", type=float, default=0.0)
    parser.add_argument('--swd', help="Seed Word Dropout (default=0.0 i.e., never drop the seed word)", type=float, default=0.0)
    parser.add_argument('--no_pretrained_emb', help="Do NOT use pre-trained word embeddings", action='store_true')
    parser.add_argument('--use_bert', help="Use BERT (base uncased) for segment embedding", action='store_true')


    # Model Specific
    parser.add_argument('--pretrained_model', help="Pre-trained model", type=str, default='')
    parser.add_argument('--attention', help="Use word attention", action='store_true')
    parser.add_argument('--fix_w_emb', help="Fix word embeddings", action='store_true')
    parser.add_argument('--fix_a_emb', help="Fix aspect embeddings", action='store_true')
    parser.add_argument('--model_type', help="Model type (embedding_based vs bow_based)", type=str, default='embedding_based')
    parser.add_argument('--deep_aspect_clf', help="Use a deep CLF on top of word embeddings", type=str, default='NO')


    args = parser.parse_args()
    args.enable_gpu = True

    seeds = [20, 7, 1993, 42, 127]
    args.seed = seeds[args.version]
    torch.cuda.manual_seed(args.seed)
    seed(args.seed)

    print('Using BERT for segment encoding')
    from bert_serving.client import BertClient

    bc = BertClient()

    all_domains = ['bags_and_cases', 'keyboards', 'boots', 'bluetooth', 'tv', 'vacuums']
    all_semeval_domains = ['english_restaurants', 'spanish_restaurants', 'french_restaurants', 'russian_restaurants',
                           'dutch_restaurants', 'turkish_restaurants']
    which_domain = all_semeval_domains



    datafolder = "/home/gkaraman/code/oposum/data/"
    savefolder = "/home/gkaraman/data1/data/bert/pretrained_embeddings_absa_semeval_multilingual/"
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)


    for domain in which_domain:
        args.domain = domain
        loggerfile = '{}_log.log'.format(domain)

        print("\n\n\n\n DOMAIN={}".format(domain))
        print("Loading train data...")
        if domain in all_domains:
            train_file = datafolder + "preprocessed/{}_MATE.hdf5".format(domain.upper())
            train_segs, train_original, train_scodes = BaselineDataHandler.load_train_segments(train_file)
        elif domain in all_semeval_domains:
            train_file = datafolder + "semeval/preprocessed/{}_TRAIN.pkl".format(domain)
            train_segs, train_original, train_scodes = BaselineDataHandler.load_train_segments_semeval(train_file)

        print("Computing segment lengths...")
        seg_lens = [len(x.split()) for x in train_original]
        sorted_ind = np.argsort(seg_lens)
        train_segs_sorted = [train_segs[i] for i in sorted_ind]
        train_original_sorted = [train_original[i] for i in sorted_ind]
        train_scodes_sorted = [train_scodes[i] for i in sorted_ind]

        joblib.dump(train_scodes_sorted,savefolder+'{}_train_scodes.pkl'.format(domain))

        train_segs = train_segs_sorted
        train_original = train_original_sorted
        train_scodes = train_scodes_sorted

        total = len(train_segs)
        total_batches = int(len(train_segs) / args.batch_size) + 1

        all_emb = []
        for i, batch_st in enumerate(range(0, total, args.batch_size)):
            print("[{}] Running BERT: {}/{}".format(domain,i,total_batches))
            batch_end = min(batch_st + args.batch_size, total)
            bert_emb = bc.encode(train_original[batch_st:batch_end])
            all_emb.append(bert_emb)

        all_emb = np.concatenate(all_emb, axis=0)
        joblib.dump(all_emb, savefolder+'{}_bert_embeddings.pkl'.format(domain))

    # GET EMBEDDINGS FOR DEV DATA
    for domain in which_domain:
        args.domain = domain
        loggerfile = '{}_log.log'.format(domain)
        print("\n\n\n\n DOMAIN={}".format(domain))

        print("Loading dev data...")
        if domain in all_domains:
            train_file = datafolder + "preprocessed/{}_MATE_DEV.hdf5".format(domain.upper())
            train_segs, train_labels, train_original, train_scodes = BaselineDataHandler.load_test_segments(train_file)
        elif domain in all_semeval_domains:
            train_file = datafolder + "semeval/preprocessed/{}_DEV.pkl".format(domain)
            train_segs, train_labels, train_original, train_scodes = BaselineDataHandler.load_test_segments_semeval(train_file)

        print("Computing segment lengths...")
        seg_lens = [len(x.split()) for x in train_original]
        sorted_ind = np.argsort(seg_lens)
        train_segs_sorted = [train_segs[i] for i in sorted_ind]
        train_original_sorted = [train_original[i] for i in sorted_ind]
        train_scodes_sorted = [train_scodes[i] for i in sorted_ind]

        joblib.dump(train_scodes_sorted,savefolder+'{}_dev_scodes.pkl'.format(domain))

        train_segs = train_segs_sorted
        train_original = train_original_sorted
        train_scodes = train_scodes_sorted

        total = len(train_segs)
        total_batches = int(len(train_segs) / args.batch_size) + 1

        all_emb = []
        for i, batch_st in enumerate(range(0, total, args.batch_size)):
            print("[{}] Running BERT: {}/{}".format(domain,i,total_batches))
            batch_end = min(batch_st + args.batch_size, total)
            bert_emb = bc.encode(train_original[batch_st:batch_end])
            all_emb.append(bert_emb)

        all_emb = np.concatenate(all_emb, axis=0)
        joblib.dump(all_emb, savefolder+'{}_bert_embeddings_dev.pkl'.format(domain))

    # GET EMBEDDINGS FOR TEST DATA
    for domain in which_domain:
        args.domain = domain
        loggerfile = '{}_log.log'.format(domain)
        print("\n\n\n\n DOMAIN={}".format(domain))
        print("Loading test data...")
        if domain in all_domains:
            train_file = datafolder + "preprocessed/{}_MATE_TEST.hdf5".format(domain.upper())
            train_segs, train_labels, train_original, train_scodes = BaselineDataHandler.load_test_segments(train_file)
        elif domain in all_semeval_domains:
            train_file = datafolder + "semeval/preprocessed/{}_TEST.pkl".format(domain)
            train_segs, train_labels, train_original, train_scodes = BaselineDataHandler.load_test_segments_semeval(train_file)


        print("Computing segment lengths...")
        seg_lens = [len(x.split()) for x in train_original]
        sorted_ind = np.argsort(seg_lens)
        train_segs_sorted = [train_segs[i] for i in sorted_ind]
        train_original_sorted = [train_original[i] for i in sorted_ind]
        train_scodes_sorted = [train_scodes[i] for i in sorted_ind]

        joblib.dump(train_scodes_sorted, savefolder + '{}_test_scodes.pkl'.format(domain))

        train_segs = train_segs_sorted
        train_original = train_original_sorted
        train_scodes = train_scodes_sorted
        # import pdb; pdb.set_trace()

        total = len(train_segs)
        total_batches = int(len(train_segs) / args.batch_size) + 1

        all_emb = []
        for i, batch_st in enumerate(range(0, total, args.batch_size)):
            print("[{}] Running BERT: {}/{}".format(domain, i, total_batches))
            batch_end = min(batch_st + args.batch_size, total)
            bert_emb = bc.encode(train_original[batch_st:batch_end])
            all_emb.append(bert_emb)

        all_emb = np.concatenate(all_emb, axis=0)
        joblib.dump(all_emb, savefolder + '{}_bert_embeddings_test.pkl'.format(domain))
