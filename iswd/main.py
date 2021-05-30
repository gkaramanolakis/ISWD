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

from DataHandler import DataHandler
from Evaluator import Evaluator
from Logger import get_logger
from model_library import AspectCLF, StudentBoWCLF, smooth_cross_entropy

all_domains = ['bags_and_cases', 'keyboards', 'boots', 'bluetooth', 'tv', 'vacuums']
all_semeval_domains = ['english_restaurants', 'spanish_restaurants', 'french_restaurants', 'russian_restaurants',
                       'dutch_restaurants', 'turkish_restaurants']


class Trainer:
    def __init__(self, args):
        self.args = args
        self.comment = '_{}'.format(args.domain)
        self.args.one_hot = True if self.args.loss in ['SmoothCrossEntropy', "KL"] else False
        self.datahandler = DataHandler(self.args)
        self.writer = SummaryWriter(log_dir=self.args.logdir)
        loggerfile = os.path.join(self.args.logdir, 'log.log')
        self.logger = get_logger(logfile=loggerfile)
        self.check_gpu()
        joblib.dump(self.args, os.path.join(self.args.logdir, 'args.pkl'))

        self.evaluator = Evaluator(args)
        if args.no_seed_weights:
            seed_weights = None
        else:
            self.logger.info('Initializing Teacher with pre-defined seed weights...')
            seed_weights = self.datahandler.seed_w

        pretrained_emb = self.datahandler.w_emb

        if self.datahandler.num_aspects != self.args.num_aspects:
            self.args.num_aspects = self.datahandler.num_aspects

        if args.model_type == 'embedding_based':
            self.logger.info('Model: Embeddings based Classifier')
            self.model = AspectCLF(vocab_size=self.datahandler.vocab_size, pretrained_emb=pretrained_emb,
                                   emb_size=self.datahandler.emb_size,
                                   seed_encodings=None, seed_weights=seed_weights, num_aspects=self.args.num_aspects,
                                   num_seeds=args.num_seeds, fix_a_emb=False, fix_w_emb=args.fix_w_emb,
                                   attention=args.attention,
                                   deep_clf=args.deep_aspect_clf, enable_gpu=args.enable_gpu,
                                   cuda_device=args.cuda_device,
                                   emb_dropout=args.emb_dropout, batch_norm=args.batch_norm, use_bert=args.use_bert,
                                   bert_model=args.bert_model)
        elif args.model_type == 'bow_based':
            self.logger.info('Model: BoW Classifier')
            self.model = StudentBoWCLF(self.datahandler.id2word, self.datahandler.aspects_ids)
        else:
            raise (BaseException('unknown model type: {}'.format(args.model_type)))

        self.model = self.cuda(self.model)
        self.optimizer = self.get_optimizer(args)
        self.loss_fn = self.get_loss_fn(args)
        self.logger.info('Saving log at {}'.format(loggerfile))
        self.epoch = -1
        self.results = []
        self.metric = self.args.target_metric
        self.best_score = -1.0
        self.best_test_score = -1.0

    def check_gpu(self):
        if self.args.enable_gpu:
            torch.cuda.manual_seed(self.args.seed)
        if self.args.enable_gpu and not torch.cuda.is_available():
            raise (BaseException('CUDA is not supported in this machine. Please rerun by setting enable_gpu=False'))
        if torch.cuda.device_count() > 1:
            self.logger.info("Tip: You could use {} GPUs in this machine!".format(torch.cuda.device_count()))

    def get_optimizer(self, args):
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'Adadelta':
            optimizer = torch.optim.Adadelta(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                        momentum=args.momentum)
        else:
            raise (NotImplementedError('unknown optimizer: {}'.format(args.optimizer)))
        return optimizer

    def get_loss_fn(self, args):
        if args.loss == 'CrossEntropy':
            loss_fn = nn.CrossEntropyLoss()
        elif args.loss == 'NLL':
            loss_fn = nn.NLLLoss()
        elif args.loss == 'SmoothCrossEntropy':
            loss_fn = smooth_cross_entropy
        elif args.loss == 'KL':
            loss_fn = nn.KLDivLoss()
        else:
            raise (NotImplementedError('unknown loss function: {}'.format(args.loss)))
        return loss_fn

    def cuda(self, x):
        if self.args.enable_gpu:
            return x.cuda(self.args.cuda_device)
        else:
            return x

    def train(self):
        # Seed word distillation: train the Student using the Teacher that is based on seed words
        self.model.train()
        all_losses = []
        all_preds = self.cuda(torch.Tensor())
        all_labels = []

        for batch in self.datahandler.get_train_batches():
            self.optimizer.zero_grad()
            i = batch['ind']
            pred = self.model(batch)
            label = batch['label']

            if self.args.loss not in ["SmoothCrossEntropy", "KL"]:
                all_labels.extend(list(label))
                label = self.cuda(Variable(torch.LongTensor(label)))
            else:
                # Convert the ground-truth aspect scores into probabilities summing to 1.
                all_labels.extend([np.argmax(l) for l in label])
                label = self.cuda(Variable(torch.Tensor(label)))
                label = F.normalize(label, p=1, dim=1)
            loss = self.loss_fn(pred, label)
            all_losses.append(loss.data.cpu().numpy())
            loss.backward()
            self.optimizer.step()

            all_preds = torch.cat((all_preds, pred.data), dim=0)

            if (self.args.report_every != -1) and (i % self.args.report_every == 0) and (i > 0):
                avg_loss = np.mean(all_losses[-self.args.report_every:])
                self.logger.debug(
                    '[{}][{}:{}/{}]\tLoss: {:f}'.format(self.args.domain, self.epoch, i, batch['total'], avg_loss))

        all_proba = all_preds.cpu().numpy()
        max_prob, all_preds = all_preds.max(dim=1)
        all_preds = all_preds.cpu().numpy()
        avg_loss = np.mean(all_losses)
        res = self.evaluator.evaluate(all_preds, all_labels, all_proba, gt_classes=range(self.args.num_aspects), verbose=False)
        res['loss'] = avg_loss
        self.epoch_results['train'] = res
        self.writer.add_histogram('train_loss{}'.format(self.comment), np.array(all_losses), self.epoch, bins=100)

    def validate(self):
        # Report validation performance after each epoch and pick the model with best validation performance
        self.model.eval()
        all_losses = []
        all_preds = self.cuda(torch.Tensor())
        all_labels = []
        for batch in self.datahandler.get_eval_batches():
            pred = self.model(batch)
            label = batch['label']
            if self.args.loss not in ["SmoothCrossEntropy", "KL"]:
                all_labels.extend(list(label))
                label = self.cuda(Variable(torch.LongTensor(label)))
            else:
                all_labels.extend(list(label))
                one_hot = np.zeros((len(label), self.args.num_aspects))
                one_hot[np.arange(len(label)), label] = 1
                label = self.cuda(Variable(torch.Tensor(one_hot)))
            loss = self.loss_fn(pred, label)
            all_losses.append(loss.data.cpu().numpy())
            all_preds = torch.cat((all_preds, pred.data), dim=0)

        all_proba = all_preds.cpu().numpy()
        max_prob, all_preds = all_preds.max(dim=1)
        all_preds = all_preds.cpu().numpy()

        avg_loss = np.mean(all_losses)
        res = self.evaluator.evaluate(all_preds, all_labels, all_proba, gt_classes=range(self.args.num_aspects), verbose=False)
        res['loss'] = avg_loss
        if res[self.metric] >= self.best_score:
            # Save the model with best validation performance
            self.best_score = res[self.metric]
            torch.save(self.model.state_dict(), os.path.join(self.args.logdir, 'best_valid_model.pt'))
        self.epoch_results['valid'] = res
        self.writer.add_histogram('valid_loss{}'.format(self.comment), np.array(all_losses), self.epoch, bins=100)
        self.flattened_valid_result_dict = self.evaluator.flattened_result_dict

    def validate_test(self):
        # Report test performance after each epoch for final plots.
        self.model.eval()
        all_losses = []
        all_preds = self.cuda(torch.Tensor())
        all_labels = []
        for batch in self.datahandler.get_test_batches():
            pred = self.model(batch)
            label = batch['label']
            if self.args.loss not in ["SmoothCrossEntropy", "KL"]:
                all_labels.extend(list(label))
                label = self.cuda(Variable(torch.LongTensor(label)))
            else:
                all_labels.extend(list(label))
                one_hot = np.zeros((len(label), self.args.num_aspects))
                one_hot[np.arange(len(label)), label] = 1
                label = self.cuda(Variable(torch.Tensor(one_hot)))
            loss = self.loss_fn(pred, label)
            all_losses.append(loss.data.cpu().numpy())

            all_preds = torch.cat((all_preds, pred.data), dim=0)

        all_proba = all_preds.cpu().numpy()
        max_prob, all_preds = all_preds.max(dim=1)
        all_preds = all_preds.cpu().numpy()

        avg_loss = np.mean(all_losses)
        res = self.evaluator.evaluate(all_preds, all_labels, all_proba, gt_classes=range(self.args.num_aspects), verbose=False)
        res['loss'] = avg_loss
        if res[self.metric] >= self.best_test_score:
            self.best_test_score = res[self.metric]
            torch.save(self.model.state_dict(), os.path.join(self.args.logdir, 'best_test_model.pt'))
        self.epoch_results['test'] = res
        self.writer.add_histogram('test_loss{}'.format(self.comment), np.array(all_losses), self.epoch, bins=100)
        self.flattened_test_result_dict = self.evaluator.flattened_result_dict

    def test(self, savename='results.pkl'):
        self.model.eval()
        all_preds = self.cuda(torch.Tensor())
        all_labels = []
        for batch in self.datahandler.get_test_batches():
            i = batch['ind']
            pred = self.model(batch)
            label = batch['label']

            all_preds = torch.cat((all_preds, pred.data), dim=0)
            all_labels.extend(list(label))

        all_proba = all_preds.cpu().numpy()
        max_prob, all_preds = all_preds.max(dim=1)
        all_preds = all_preds.cpu().numpy()
        res = self.evaluator.evaluate(all_preds, all_labels, all_proba, gt_classes=range(self.args.num_aspects), verbose=False)
        self.epoch_results['test'] = res
        self.logger.info('Test {}: {:.3}'.format(self.metric, res[self.metric]))
        self.logger.info('Confusion Matrix:\n{}'.format(res['conf_mat']))
        joblib.dump(res, os.path.join(self.args.logdir, savename))

    def start_epoch(self):
        # Do necessary stuff at the beginning of each epoch
        self.epoch_results = {}
        return

    def end_epoch(self):
        # Do necessary stuff at the end of each epoch
        self.writer.add_scalars('loss{}'.format(self.comment), {
            'train_loss': self.epoch_results['train']['loss'],
            'valid_loss': self.epoch_results['valid']['loss']}, self.epoch)
        score = self.epoch_results['valid'][self.metric]
        test_score = self.epoch_results['test'][self.metric]
        self.logger.info('[{}] {} (dev): {:.3}'.format(self.args.domain, self.metric, score))
        self.logger.info('[{}] {} (test): {:.3}'.format(self.args.domain, self.metric, test_score))
        self.writer.add_scalars(self.metric, {self.args.domain: score}, self.epoch)
        self.writer.add_scalars('test_' + self.metric, {self.args.domain: score}, self.epoch)

        res_flattened = self.flattened_test_result_dict
        res_flattened['avg_prec'] = np.average(self.epoch_results['valid']['prec'])
        res_flattened['avg_rec'] = np.average(self.epoch_results['valid']['rec'])
        important_list = ['acc', 'avg_prec', 'avg_rec', 'macro_average_f1', 'micro_average_f1']
        self.writer.add_scalars('average_test_results{}'.format(self.comment),
                                {x: res_flattened[x] for x in important_list}, self.epoch)
        self.writer.add_scalars('test_results{}'.format(self.comment),
                                {x: res_flattened[x] for x in res_flattened if not 'conf' in x}, self.epoch)
        self.writer.add_scalars('test_conf_matrix{}'.format(self.comment),
                                {x: res_flattened[x] for x in res_flattened if 'conf' in x}, self.epoch)

        self.results.append(self.epoch_results)
        joblib.dump(self.results, os.path.join(self.args.logdir, 'epoch_results.pkl'))  # saving intermediate results
        return

    def close(self):
        self.writer.close()
        torch.save(self.model.state_dict(), os.path.join(self.args.logdir, 'last_model.pt'))
        joblib.dump(self.results, os.path.join(self.args.logdir, 'results.pkl'))
        self.logger.info("Process ended in {:.3f} s".format(self.total_time))
        self.logger.info("Results stored at {}".format(self.args.logdir))

    def process(self):
        self.total_time = 0
        for epoch in range(self.args.num_epochs):
            t0 = time()
            self.epoch = epoch
            self.start_epoch()
            self.train()
            self.validate()
            self.validate_test()
            epoch_time = time() - t0
            self.total_time += epoch_time
            self.logger.info("Epoch {} Done in {} s.".format(self.epoch, epoch_time))
            self.epoch_results['time'] = epoch_time
            self.end_epoch()
        self.test()
        self.close()


def report_results(logdir, domains, metric='micro_average_f1'):
    print("\n\n\t\tAggregating Results")
    if not os.path.exists(logdir):
        print("[ERROR] logdir not existent: {}".format(logdir))

    with open(logdir + '/results.txt', 'w') as f:
        def print_and_write(text):
            print(text)
            f.write(text + '\n')

        results = {}
        best_epoch = {}
        best_res = {}
        for domain in domains:
            print("Loading results for {}".format(domain))
            res_file = logdir + '/' + domain + '/' + 'epoch_results.pkl'
            if not os.path.exists(res_file):
                print("[ERROR] Could not find {}".format(res_file))
                continue
            results[domain] = joblib.load(res_file)
            all_scores = [x['valid'][metric] for x in results[domain]]
            best_epoch[domain] = np.argmax(all_scores)
            best_res[domain] = results[domain][best_epoch[domain]]

        print_and_write("Results Summary: ")
        best_scores = [best_res[x]['test'][metric] if x in best_res else -1 for x in domains]
        print_and_write("Domain: {}".format("\t".join(domains) + "\tAVG"))
        print_and_write(
            "\t" + "\t".join(["{:.3}".format(x) for x in best_scores] + ["{:.3}".format(np.average(best_scores))]))
        print_and_write("\t" + "\t".join([str(best_epoch[x]) if x in best_epoch else -1 for x in domains] + ['-1']))
    return


def print_test_results(logdir, domains, f):
    if not os.path.exists(logdir):
        print("[ERROR] logdir not existent: {}".format(logdir))

    def print_and_write(text, n=True):
        if n:
            print(text)
            f.write(text + "\n")
        else:
            print(text),
            f.write(text)

    results = {}
    selected_epoch = {}
    best_epoch = {}
    res = {}
    best_res = {}
    for domain in domains:
        res_file = logdir + '/' + domain + '/' + 'results.pkl'
        if not os.path.exists(res_file):
            print('Error: not existent file: {}'.format(res_file))
            return -1, -1
        results[domain] = joblib.load(res_file)
        all_scores = [x['valid']['micro_average_f1'] for x in results[domain]]
        all_test_scores = [x['test']['micro_average_f1'] for x in results[domain]]
        selected_epoch[domain] = np.argmax(all_scores)
        best_epoch[domain] = np.argmax(all_test_scores)
        res[domain] = results[domain][selected_epoch[domain]]
        best_res[domain] = results[domain][best_epoch[domain]]

    scores = [res[x]['test']['micro_average_f1'] if x in res else -1 for x in domains]
    best_scores = [best_res[x]['test']['micro_average_f1'] if x in res else -1 for x in domains]

    for i, domain in enumerate(domains):
        print_and_write("{:.1f}({}),{:.1f}({}) ".format(100 * scores[i], selected_epoch[domain],
                                                        100 * best_scores[i], best_epoch[domain]), n=False)
    avg_score = np.average(scores)
    best_avg_score = np.average(best_scores)

    print_and_write("{:.1f},{:.1f}  ".format(100 * avg_score, 100 * best_avg_score))
    scores.append(avg_score)
    best_scores.append(best_avg_score)
    return scores, best_scores


def report_avg_results(logdir, domains=all_domains):
    # Reports the average results when having ran each experiment for 5 times.
    mainfolder = "/".join(logdir.split('/')[0:-1]) + '/'
    logdir = logdir.split('/')[-1]
    basefolder = mainfolder + logdir + '/'
    writefile = basefolder + '/res.txt'
    f = open(writefile, 'w')

    def print_and_write(text, n=True):
        if n:
            print(text)
            f.write(text + "\n")
        else:
            print(text),
            f.write(text)

    print_and_write('parsing {}\n'.format(basefolder))
    all_scores = []
    all_best_scores = []
    for fpath in glob.glob(basefolder + '*/'):

        s, bs = print_test_results(fpath, domains, f)
        if s == -1:
            return -1, -1
        all_scores.append(s)
        all_best_scores.append(bs)
    all_scores = np.array(all_scores)
    avg_scores = np.mean(all_scores, axis=0)
    std_scores = np.std(all_scores, axis=0)
    print_and_write("\n")
    print_and_write("AVERAGE:        ", n=False)
    for i in range(7):
        print_and_write("{:.1f} ({:.1f}),".format(100 * avg_scores[i], 100 * std_scores[i]), n=False)
    print

    print_and_write("\nREPORT:")
    report_str = ""
    for i in range(7):
        ps = "{:.1f},".format(100 * avg_scores[i])
        print_and_write(ps, n=False)
        report_str += ps
    ps = "{}".format(basefolder)
    ps_small = "{}".format(logdir.split('/')[-1])
    print_and_write(ps_small + "," + ps)
    report_str += (ps_small + "," + ps)
    print("\n\n")
    return report_str


def run_trainer(args, domain):
    print("Running {}".format(domain))
    args.domain = domain

    # Define output paths
    args.logdir += '/' + domain + '/'
    trainer = Trainer(args)
    trainer.process()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('domain', help="Domain name (without extension)", type=str, default='pairs')

    # Trainer Specific
    parser.add_argument('--logdir', help="log directory for tensorboard", type=str, default='../experiments')
    parser.add_argument('--debug', help="Enable debug mode", action='store_true')
    parser.add_argument('--num_epochs', help="Number of epochs (default: 25)", type=int, default=25)
    parser.add_argument('--loss', help="Loss Function (CrossEntropy / NLL)", type=str, default='CrossEntropy')
    parser.add_argument('--optimizer', help="Optimizer (Adam / Adadelta)", type=str, default='Adam')
    parser.add_argument('--lr', help="Learning rate (default: 0.0001)", type=float, default=0.0001)
    parser.add_argument('--weight_decay', help="Weight Decay", type=float, default=0.0)
    parser.add_argument('--momentum', help="Momentum (used for optimizer=SGD)", type=float, default=0.9)
    # parser.add_argument('--seed', help="Random seed (default: 42)", type=int, default=42)
    parser.add_argument('--report_every', help="Report every x number of batches", type=int, default=50)
    parser.add_argument('--cuda_device', help="CUDA Device ID", type=int, default=0)
    parser.add_argument('--batch_size', help="Batch Size", type=int, default=1024)
    parser.add_argument('--target_metric', help="Target Metric to report", type=str, default='micro_average_f1')
    parser.add_argument('--version', help="Run # (1..5)", type=int, default=1)
    parser.add_argument('--disable_gpu', help="Disable GPU", action='store_true')

    # Domain Specific
    parser.add_argument('--test_data', help="hdf5 file of test segments", type=str, default='')
    parser.add_argument('--min_len', help="Minimum number of non-stop-words in segment (default: 2)", type=int,
                        default=2)
    parser.add_argument('--num_aspects', help="Number of aspects (default: 9)", type=int, default=9)
    parser.add_argument('--aspect_seeds', help='file that contains aspect seed words (overrides number of aspects)',
                        type=str, default='')
    parser.add_argument('-q', '--quiet', help="No information to stdout", action='store_true')
    parser.add_argument('--num_seeds', help="Number of seed words to use (default: 30)", type=int, default=30)
    parser.add_argument('--no_seed_weights', help="Forcing the *unweighted* avg of seed word embeddings",
                        action='store_true')
    parser.add_argument('--batch_norm', help="Batch normalization on segment encodings", action='store_true')
    parser.add_argument('--emb_dropout', help="Dropout at the segment embedding layer", type=float, default=0.0)
    parser.add_argument('--swd', help="Seed Word Dropout (default=0.0 i.e., never drop the seed word)", type=float,
                        default=0.0)
    parser.add_argument('--no_pretrained_emb', help="Do NOT use pre-trained word embeddings", action='store_true')
    parser.add_argument('--use_bert', help="Use BERT (base uncased) for segment embedding", action='store_true')
    parser.add_argument('--bert_model', help="Type of BERT model: base/large", type=str, default='base')

    # Model Specific
    parser.add_argument('--pretrained_model', help="Pre-trained model", type=str, default='')
    parser.add_argument('--attention', help="Use word attention", action='store_true')
    parser.add_argument('--fix_w_emb', help="Fix word embeddings", action='store_true')
    parser.add_argument('--fix_a_emb', help="Fix aspect embeddings", action='store_true')
    parser.add_argument('--model_type', help="Model type (embedding_based vs bow_based)", type=str,
                        default='embedding_based')
    parser.add_argument('--deep_aspect_clf', help="Use a deep CLF on top of word embeddings", type=str, default='NO')

    args = parser.parse_args()
    args.enable_gpu = not args.disable_gpu

    seeds = [20, 7, 1993, 42, 127]
    args.seed = seeds[args.version]
    torch.cuda.manual_seed(args.seed)
    seed(args.seed)


    if args.debug:
        args.logdir = './debug'
        if os.path.exists(args.logdir):
            os.system('rm -rf {}'.format(args.logdir))
    else:
        args.logdir = args.logdir + \
                      "_numseeds{}".format(args.num_seeds) + \
                      "_lr{}".format(args.lr) + \
                      "_{}".format(args.loss)

        if args.emb_dropout > 0:
            args.logdir += "_embdropout{}".format(args.emb_dropout)
        if args.weight_decay > 0:
            args.logdir += "_wdecay{}".format(args.weight_decay)
        if args.deep_aspect_clf != "NO":
            args.logdir += "_{}".format(args.deep_aspect_clf)
        if args.optimizer != "":
            args.logdir += "_{}".format(args.optimizer)

        args.logdir += "_lr{}".format(args.lr)

    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    original_logdir = args.logdir
    args.logdir += '/v{}'.format(args.version)
    os.mkdir(args.logdir)
    print('\t\tEXPERIMENT with domain={}\nargs: {}\nlogdir: {}'.format(args.domain, args, args.logdir))

    if args.domain == 'all':
        # Run multiple processes, one for each dataset
        from multiprocessing import Pool
        from functools import partial

        domains = all_domains
        new_fun = partial(run_trainer, args)
        p = Pool(len(domains))
        p.map(new_fun, domains)
        p.close()
        torch.cuda.empty_cache()
        report_results(args.logdir, domains, metric=args.target_metric)
    elif args.domain == 'pairs':
        from multiprocessing import Pool
        from functools import partial

        domains = all_domains[0:2]
        new_fun = partial(run_trainer, args)
        p = Pool(len(domains))
        p.map(new_fun, domains)
        p.close()
        torch.cuda.empty_cache()

        domains = all_domains[2:4]
        new_fun = partial(run_trainer, args)
        p = Pool(len(domains))
        p.map(new_fun, domains)
        p.close()
        torch.cuda.empty_cache()

        domains = all_domains[4:6]
        new_fun = partial(run_trainer, args)
        p = Pool(len(domains))
        p.map(new_fun, domains)
        p.close()
        torch.cuda.empty_cache()

        report_results(args.logdir, all_domains, metric=args.target_metric)
        if args.version == 4:
            report_avg_results(original_logdir, all_domains)
    elif args.domain == 'semeval':
        from multiprocessing import Pool
        from functools import partial

        domains = all_semeval_domains[0:2]
        new_fun = partial(run_trainer, args)
        p = Pool(len(domains))
        p.map(new_fun, domains)
        p.close()
        torch.cuda.empty_cache()

        domains = all_semeval_domains[2:4]
        new_fun = partial(run_trainer, args)
        p = Pool(len(domains))
        p.map(new_fun, domains)
        p.close()
        torch.cuda.empty_cache()

        domains = all_semeval_domains[4:6]
        new_fun = partial(run_trainer, args)
        p = Pool(len(domains))
        p.map(new_fun, domains)
        p.close()
        torch.cuda.empty_cache()

        report_results(args.logdir, all_semeval_domains, metric=args.target_metric)
        if args.version == 4:
            report_avg_results(original_logdir, all_semeval_domains)
    elif args.domain == 'sequential':
        for domain in all_domains:
            run_trainer(args, domain)
        report_results(args.logdir, all_domains, metric=args.target_metric)
    else:
        run_trainer(args, args.domain)
