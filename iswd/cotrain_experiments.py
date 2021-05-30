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
from copy import deepcopy


from DataHandler import DataHandler

sys.path.append(os.path.join(home, "code/research_code/Spring_2018/TextModules/"))
from Evaluator import Evaluator
from Logger import get_logger
from model_library import AspectCLF, StudentBoWCLF, SeedCLF, smooth_cross_entropy

all_semeval_domains = ['english_restaurants', 'spanish_restaurants', 'french_restaurants', 'russian_restaurants',
                       'dutch_restaurants', 'turkish_restaurants']
all_domains = ['bags_and_cases', 'keyboards', 'boots', 'bluetooth', 'tv', 'vacuums']

class Trainer:
    def __init__(self, args):
        self.args = args
        self.comment = '_{}'.format(args.domain)
        if self.args.loss in ['SmoothCrossEntropy', "KL"]:
            self.args.one_hot = True
        else:
            self.args.one_hot = False
        self.datahandler = DataHandler(self.args)
        self.writer = SummaryWriter(log_dir=self.args.logdir)
        loggerfile = os.path.join(self.args.logdir, 'log.log')
        self.logger = get_logger(logfile=loggerfile)
        self.check_gpu()
        joblib.dump(self.args, os.path.join(self.args.logdir, 'args.pkl'))

        self.evaluator = Evaluator(args) 
        if args.no_seed_weights:
            self.logger.info('NOT using seed weights...')
            seed_weights = None
        else:
            self.logger.info('USING seed weights...')
            seed_weights = self.datahandler.seed_w

        if args.no_pretrained_emb:
            self.logger.info('NOT using pretrained word embeddings...')
            pretrained_emb = None
        else:
            pretrained_emb = self.datahandler.w_emb

        if self.datahandler.num_aspects != self.args.num_aspects:
            self.logger.info("Automatically changing num_aspects from {} to {}".format(self.args.num_aspects, self.datahandler.num_aspects))
            self.args.num_aspects = self.datahandler.num_aspects

        if args.model_type == 'embedding_based':
            self.logger.info('Model: Embeddings based Classifier')
            # prev model is loaded just to gather previous predictions and regularize the new model to
            # provide similar predictions.
            if args.memory_reg > 0:
                self.prev_model = AspectCLF(vocab_size=self.datahandler.vocab_size, pretrained_emb=pretrained_emb, emb_size=self.datahandler.emb_size,
                                       seed_encodings=None, seed_weights=seed_weights, num_aspects=self.args.num_aspects,
                                       num_seeds=args.num_seeds, fix_a_emb=False, fix_w_emb=args.fix_w_emb, attention=args.attention,
                                       deep_clf=args.deep_aspect_clf, enable_gpu=args.enable_gpu, cuda_device=args.cuda_device,
                                       emb_dropout=args.emb_dropout, batch_norm= args.batch_norm, use_bert=args.use_bert,
                                            bert_model=args.bert_model)
            self.model = AspectCLF(vocab_size=self.datahandler.vocab_size, pretrained_emb=pretrained_emb,  emb_size=self.datahandler.emb_size,
                                   seed_encodings=None, seed_weights=seed_weights, num_aspects=self.args.num_aspects,
                                   num_seeds=args.num_seeds, fix_a_emb=False, fix_w_emb=args.fix_w_emb, attention=args.attention,
                                   deep_clf=args.deep_aspect_clf, enable_gpu=args.enable_gpu, cuda_device=args.cuda_device,
                                   emb_dropout=args.emb_dropout, batch_norm= args.batch_norm, use_bert=args.use_bert,
                                   bert_model=args.bert_model)
        elif args.model_type == 'bow_based':
            self.logger.info('Model: BoW Classifier')
            self.model = StudentBoWCLF(self.datahandler.id2word, self.datahandler.aspects_ids)
        else:
            raise(BaseException('unknown model type: {}'.format(args.model_type)))
        self.model = self.cuda(self.model)

        self.teacher = SeedCLF(self.datahandler.id2word, self.datahandler.aspects_ids, seed_weights,
                               verbose=0, general_ind=self.datahandler.general_ind,
                               hard_pred=args.hard_teacher_pred)

        self.optimizer = self.get_optimizer(args)
        if args.scheduler_gamma > 0:
            ms=args.bootstrap_epoch
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[ms, ms+1, ms+2, ms+3], gamma=args.scheduler_gamma)
        self.loss_fn = self.get_loss_fn(args)
        self.logger.info('Saving log at {}'.format(loggerfile))
        self.logger.debug('enable_gpu={}'.format(args.enable_gpu))
        self.epoch = -1
        self.results = []
        self.metric = self.args.target_metric
        self.best_score = -1.0
        self.best_test_score = -1.0
        self.epoch_results = {}
        if args.memory_reg > 0:
            self.memory_loss = self.get_memory_loss_fn(args)
            self.prev_model = self.cuda(self.prev_model)

        self.student_proba_train = None
        self.student_proba_dev = None
        self.student_proba_test  = None
        self.labels_dev = None
        self.labels_test = None
        self.teacher_proba_train = None
        self.teacher_pred_dev = None
        self.teacher_pred_test = None
        self.disagreement = -1

    def check_gpu(self):
        if self.args.enable_gpu:
            torch.cuda.manual_seed(self.args.seed)
        if self.args.enable_gpu and not torch.cuda.is_available():
            raise(BaseException('CUDA is not supported in this machine. Please rerun by setting enable_gpu=False'))
        if torch.cuda.device_count() > 1:
            self.logger.info("Tip: You could use {} GPUs in this machine!".format(torch.cuda.device_count()))

    def get_optimizer(self, args):
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'Adadelta':
            optimizer = torch.optim.Adadelta(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise(NotImplementedError('unknown optimizer: {}'.format(args.optimizer)))
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
            raise(NotImplementedError('unknown loss function: {}'.format(args.loss)))
        return loss_fn


    def get_memory_loss_fn(self, args):
        if args.memory_loss == 'CrossEntropy':
            loss_fn = nn.CrossEntropyLoss()
        elif args.memory_loss == 'NLL':
            loss_fn = nn.NLLLoss()
        elif args.memory_loss == 'SmoothCrossEntropy':
            loss_fn = smooth_cross_entropy
        elif args.memory_loss == 'KL':
            loss_fn = nn.KLDivLoss()
        else:
            raise(NotImplementedError('unknown loss function: {}'.format(args.loss)))
        return loss_fn

    def cuda(self, x):
        if self.args.enable_gpu:
            return x.cuda(self.args.cuda_device)
        else:
            return x

    def train(self):
        self.model.train()
        if self.args.memory_reg > 0:
            self.prev_model.eval()
        all_losses = []
        all_memory_losses = []
        all_preds = self.cuda(torch.Tensor())
        all_labels = []
        self.teacher_scores_train = []
        self.teacher_seed_word_pred_train = []

        if args.scheduler_gamma > 0:
            self.scheduler.step()
            self.logger.info("Optimizing with lr={}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))

        if self.teacher.conf_mat is None:
            self.logger.info("TEACHER does NOT use confusion matrices")
        else:
            self.logger.info("TEACHER uses confusion matrices")
        for batch in self.datahandler.get_train_batches():
            self.optimizer.zero_grad()
            i = batch['ind']
            if (args.deep_aspect_clf in ['CNN', 'charCNN']) and i > batch['total'] - 20: #or
                # ignore really big segments when clf is CNN to avoid OOM error. 
                break

            pred = self.model(batch)

            # I use different ids for the teacher, because if SWD=1, then the seed words are dropped from batch['ids']. 
            teacher_scores, teacher_seed_word_pred = map(list, zip(*[self.teacher.predict_verbose(seg) for seg in batch['teacher_ids'].tolist()]))

            if self.args.loss not in ["SmoothCrossEntropy", "KL"]:
                label = np.argmax(teacher_scores, axis=1)
                all_labels.extend(list(label))
                label = self.cuda(Variable(torch.LongTensor(label)))
            else:
                # Convert the ground-truth aspect scores into probabilities summing to 1.
                label = teacher_scores
                all_labels.extend([np.argmax(l) for l in label])
                label = self.cuda(Variable(torch.Tensor(label)))
                label = F.softmax(label, dim=1)
            loss = self.loss_fn(pred, label)
            all_losses.append(loss.data.cpu().numpy())

            if args.memory_reg == 0.0:
                loss.backward()
            else:
                # Regularize the model to avoid forgetting the previous weights / predictions.
                prev_pred = F.softmax(self.prev_model(batch), dim=1)
                memory_loss = self.memory_loss(pred, prev_pred)
                all_memory_losses.append(memory_loss.data.cpu().numpy())
                total_loss = (1 - args.memory_reg) * loss + args.memory_reg * memory_loss
                loss += memory_loss
                total_loss.backward()

            self.optimizer.step()

            all_preds = torch.cat((all_preds, pred.data), dim=0)
            self.teacher_scores_train.extend(teacher_scores)
            self.teacher_seed_word_pred_train.extend(teacher_seed_word_pred)

            if (self.args.report_every != -1) and (i % self.args.report_every == 0) and (i > 0):
                avg_loss = np.mean(all_losses[-self.args.report_every:])
                avg_memory_loss = np.mean(all_memory_losses[-self.args.report_every:])
                if args.memory_reg == 0:
                    self.logger.debug('[{}][{}:{}/{}]\tLoss: {:f}'.format(self.args.domain, self.epoch, i, batch['total'], avg_loss))
                else:
                    self.logger.debug('[{}][{}:{}/{}]\tLoss: {:.5f}\tMemory Loss: {:.5f}'.format(self.args.domain, self.epoch, i, batch['total'], avg_loss, avg_memory_loss))

        all_proba = all_preds.cpu().numpy()
        self.student_proba_train = all_proba
        max_prob, all_preds = all_preds.max(dim=1)
        all_preds = all_preds.cpu().numpy()
        avg_loss = np.mean(all_losses)
        res = self.evaluator.evaluate_group(all_preds, all_labels, all_proba, gt_classes=range(self.args.num_aspects),verbose=False)
        res['loss'] = avg_loss
        self.epoch_results['train'] = res
        self.writer.add_histogram('train_loss{}'.format(self.comment), np.array(all_losses), self.epoch, bins=100)

        # save disagreement
        s_pred_hard = np.argmax(self.student_proba_train, axis=1)
        t_pred_hard = np.argmax(self.teacher_scores_train, axis=1)
        self.disagreement =  ((s_pred_hard != t_pred_hard).sum()) / float(s_pred_hard.shape[0])
        self.epoch_results['hard_disagreement'] = self.disagreement


    def update_teacher(self):
        # Use Maximum Likelihood Estimation to update the seed word confusion matrices.
        assert self.student_proba_train is not None, "Student proba is None."
        assert self.teacher_scores_train is not None, "Teacher scores is None."


        s_pred_hard = np.argmax(self.student_proba_train, axis=1)
        s_pred_soft = F.softmax(torch.Tensor(self.student_proba_train), dim=1).numpy()
        t_pred_hard = np.argmax(self.teacher_scores_train, axis=1)
        seed_word_occurences = np.array(self.teacher_seed_word_pred_train)
        teacher_answers = seed_word_occurences.sum(axis=1) > 0
        self.disagreement =  ((s_pred_hard[teacher_answers] != t_pred_hard[teacher_answers]).sum()) / float(teacher_answers.sum())
        self.epoch_results['train_disagreement'] = self.disagreement

        K = self.args.num_aspects
        N = s_pred_hard.shape[0]

        # Initialize a zero confusion matrix for each seed word.
        conf_mat = {wid: np.zeros(K) for wid in self.teacher.seed_list}

        # Maximum Likelihood Estimation for the class priors
        self.q = np.array([np.sum(s_pred_hard == i) for i in range(K)]) / float(N)
        self.logger.info('Estimated class priors: {}'.format(",".join(["{:.2f}".format(x) for x in self.q])))

        # Maximum Likelihood Estimation for each confusion matrix
        for wid_i, wid in enumerate(self.teacher.seed_list):
            # keep the segments where this seed word has been activated
            relevant_ind = (seed_word_occurences[:, wid_i] > 0)
            pred_aspect = self.teacher.seed_dict[wid][0]

            if args.teacher_type == 'v1':
                # Precision-based updates
                if args.soft_updates == False:
                    conf_mat[wid] = np.array([np.sum(s_pred_hard[relevant_ind]==i) / float(np.sum(relevant_ind)) for i in range(K)])
                else:
                    conf_mat[wid] = np.array([s_pred_soft[relevant_ind][:, i].sum() for i in range(K)])
                    conf_mat[wid] = conf_mat[wid] / float(conf_mat[wid].sum())
            elif args.teacher_type == 'v2':
                # Dawid-Skene model where each seed word is applied when it occurs in the segment
                # We allow positive mass to other aspects.
                conf_mat[wid][:] = self.args.pos_mass / float(K - 1)
                conf_mat[wid][pred_aspect] = 1 - self.args.pos_mass

                student_sum = s_pred_soft[relevant_ind].sum(axis=0)  # adding student probabilities for all classes for all relevant samples
                conf_mat[wid] *= student_sum
                conf_mat[wid] /= conf_mat[wid].sum()
            else:
                raise(BaseException('{} not implemented'.format(args.teacher_type)))        

            # GRADIENT EM
            prev_param = np.zeros(K)
            prev_param[pred_aspect] = 1
            conf_mat[wid] = self.args.teacher_memory * prev_param + (1 - self.args.teacher_memory) * conf_mat[wid]  # (self.conf_mat[wid] + prev_param) / 2.0

        self.logger.info("Teacher answers on the {}% ({}/{}) of the training set".format(100 * teacher_answers.sum() / teacher_answers.shape[0], teacher_answers.sum(), teacher_answers.shape[0]))
        self.logger.info("Student-Teacher disagreement: {}/{} ({:.2f}%)".format((s_pred_hard[teacher_answers] != t_pred_hard[teacher_answers]).sum(), teacher_answers.sum(),100*self.disagreement))
        self.logger.info("Avg of seed word occurences in training set: {:.2f}".format(np.average(seed_word_occurences.sum(axis=0))))

        self.conf_mat = conf_mat
        joblib.dump(self.conf_mat, self.args.logdir + 'conf_mat_{}.pkl'.format(self.epoch))
        joblib.dump(self.q, self.args.logdir + 'prior_{}.pkl'.format(self.epoch))

        return

    def validate(self):
        self.model.eval()
        all_losses = []
        all_preds = self.cuda(torch.Tensor())
        all_labels = []
        for batch in self.datahandler.get_eval_batches():
            i = batch['ind']
            pred = self.model(batch)
            label = batch['label']
            # import pdb; pdb.set_trace()
            if self.args.loss not in ["SmoothCrossEntropy", "KL"]:
                all_labels.extend(list(label))
                label = self.cuda(Variable(torch.LongTensor(label)))
            else:
                # Convert the ground-truth label into a one-hot label and treat is as a prob distribution
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
        res = self.evaluator.evaluate_group(all_preds, all_labels, all_proba, gt_classes=range(self.args.num_aspects), verbose=False)
        res['loss'] = avg_loss
        if res[self.metric] >= self.best_score:
            # Save the best validation model
            self.best_score = res[self.metric]
            torch.save(self.model.state_dict(), os.path.join(self.args.logdir, 'best_valid_model.pt'))
        self.epoch_results['valid'] = res
        self.writer.add_histogram('valid_loss{}'.format(self.comment), np.array(all_losses), self.epoch, bins=100)
        self.flattened_valid_result_dict = self.evaluator.flattened_result_dict


    def validate_test(self):
        # Giannis: also validate on the test set
        self.model.eval()
        all_losses = []
        all_preds = self.cuda(torch.Tensor())
        all_labels = []
        for batch in self.datahandler.get_test_batches():
            i = batch['ind']
            pred = self.model(batch)
            label = batch['label']
            # import pdb; pdb.set_trace()
            if self.args.loss not in ["SmoothCrossEntropy", "KL"]:
                all_labels.extend(list(label))
                label = self.cuda(Variable(torch.LongTensor(label)))
            else:
                # Convert the ground-truth label into a one-hot label and treat is as a prob distribution
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
        res = self.evaluator.evaluate_group(all_preds, all_labels, all_proba, gt_classes=range(self.args.num_aspects), verbose=False)
        res['loss'] = avg_loss
        if res[self.metric] >= self.best_test_score:
            # Save the best test model
            self.best_test_score = res[self.metric]
            torch.save(self.model.state_dict(), os.path.join(self.args.logdir, 'best_test_model.pt'))
        self.epoch_results['test'] = res
        self.writer.add_histogram('test_loss{}'.format(self.comment), np.array(all_losses), self.epoch, bins=100)
        self.flattened_test_result_dict = self.evaluator.flattened_result_dict


    def test(self, savename='results.pkl'):
        self.model.eval()
        all_preds = self.cuda(torch.Tensor())
        all_labels = []
        teacher_scores_test = []

        for batch in self.datahandler.get_test_batches():
            i = batch['ind']
            pred = self.model(batch)
            teacher_scores, teacher_seed_word_pred = map(list, zip(*[self.teacher.predict_verbose(seg) for seg in batch['ids'].tolist()]))
            label = batch['label']

            all_preds = torch.cat((all_preds, pred.data), dim=0)
            teacher_scores_test.extend(teacher_scores)
            all_labels.extend(list(label))

        all_proba = all_preds.cpu().numpy()
        max_prob, all_preds = all_preds.max(dim=1)
        all_preds = all_preds.cpu().numpy()

        res = self.evaluator.evaluate_group(all_preds, all_labels, all_proba, gt_classes=range(self.args.num_aspects),verbose=False)
        self.epoch_results['test'] = res


        teacher_scores_test = np.array(teacher_scores_test)
        teacher_preds = np.argmax(teacher_scores_test, axis=1)
        teacher_res = self.evaluator.evaluate_group(teacher_preds, all_labels, teacher_scores_test, gt_classes=range(self.args.num_aspects), verbose=False)
        self.epoch_results['teacher_test'] = teacher_res

        self.logger.info('Test {}:\t STUDENT={:.3}\t TEACHER={:.3}'.format(self.metric, res[self.metric], teacher_res[self.metric]))
        self.logger.info('Train disagreement: {}%'.format(100*self.disagreement))
        self.logger.info('STUDENT confusion Matrix:\n{}'.format(res['conf_mat']))
        self.logger.info('TEACHER confusion Matrix:\n{}'.format(teacher_res['conf_mat']))


        joblib.dump(res, os.path.join(self.args.logdir, savename))


    def start_epoch(self):
        # Do necessary staff at the beginning of each epoch
        self.epoch_results = {}
        return

    def end_epoch(self):
        # Do necessary staff at the end of each epoch
        self.writer.add_scalars('loss{}'.format(self.comment), {
            'train_loss': self.epoch_results['train']['loss'],
            'valid_loss': self.epoch_results['valid']['loss']}, self.epoch)
        score = self.epoch_results['valid'][self.metric]
        test_score = self.epoch_results['test'][self.metric]
        self.logger.info('{}: {:.3}'.format(self.metric, score))
        self.logger.info('{} (test): {:.3}'.format(self.metric, test_score))
        self.writer.add_scalars(self.metric, {self.args.domain: score}, self.epoch)
        self.writer.add_scalars('test_' + self.metric, {self.args.domain: score}, self.epoch)

        res_flattened = self.flattened_test_result_dict
        res_flattened['avg_prec'] = np.average(self.epoch_results['valid']['prec'])
        res_flattened['avg_rec'] = np.average(self.epoch_results['valid']['rec'])
        important_list = ['acc', 'avg_prec', 'avg_rec', 'macro_average_f1', 'micro_average_f1']
        self.writer.add_scalars('average_test_results{}'.format(self.comment), {x: res_flattened[x] for x in important_list}, self.epoch)
        self.writer.add_scalars('test_results{}'.format(self.comment), {x:res_flattened[x] for x in res_flattened if not 'conf' in x}, self.epoch)
        self.writer.add_scalars('test_conf_matrix{}'.format(self.comment), {x: res_flattened[x] for x in res_flattened if 'conf' in x}, self.epoch)

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
        self.test()

        for epoch in range(self.args.num_epochs):
            if epoch == 0:
                # Do not regularize the model in the first epochs until we start bootstrapping.
                mem_reg = self.args.memory_reg
                self.args.memory_reg = 0

                # Use CrossEntropyLoss with hard targets for the first epochs.
                target_loss_fn = self.args.loss
            elif epoch == self.args.bootstrap_epoch + 1:
                # When we're done with the burnout epochs, we restore the right cotraining parameters. 
                if mem_reg > 0:
                    self.logger.info("Adding prev_model regularization with mem_reg={}".format(mem_reg))
                    self.args.memory_reg = mem_reg
                    self.prev_model.load_state_dict(deepcopy(self.model.state_dict()))
                self.logger.info("Switching to loss={}".format(target_loss_fn))
                self.args.loss = target_loss_fn
                self.loss_fn = self.get_loss_fn(self.args)
            t0 = time()
            self.epoch = epoch
            self.start_epoch()

            self.train()

            if epoch >= self.args.bootstrap_epoch:
                self.update_teacher()
                if not args.fix_teacher:
                    self.teacher.conf_mat = self.conf_mat
                    self.teacher.prior = self.q

            self.validate()
            self.validate_test()
            epoch_time = time() - t0
            self.total_time += epoch_time
            self.logger.info("Epoch {} Done in {} s.".format(self.epoch, epoch_time))
            self.epoch_results['time'] = epoch_time
            self.test()
            self.end_epoch()

        self.test()
        self.close()


def run_cotrain(args, domain):
    print("Running {}".format(domain))
    args.domain = domain

    # Define output paths
    args.logdir += '/' + domain + '/'
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    args.pretrained_model += '/' + domain + '/'
    args.student_folder = args.logdir + \
                          'student' + \
                          '_{}'.format(args.loss) + \
                          '_lr{}'.format(args.lr) + \
                          '_memloss{}'.format(args.memory_loss) + \
                          '_memreg{}'.format(args.memory_reg)

    args.teacher_folder = args.logdir + \
                          'teacher' + \
                          "_{}".format(args.teacher_type) + \
                          "_memory{}".format(args.teacher_memory)

    if not os.path.exists(args.student_folder):
        os.mkdir(args.student_folder)
    if not os.path.exists(args.teacher_folder):
        os.mkdir(args.teacher_folder)


    trainer = Trainer(args)
    trainer.process()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('domain', help="Domain name (without extension)", type=str, default='pairs')

    # Trainer Specific
    parser.add_argument('--logdir', help="log directory for tensorboard", type=str, default='../experiments/')
    parser.add_argument('--debug', help="Enable debug mode", action='store_true')
    parser.add_argument('--num_epochs', help="Number of epochs (default: 25)", type=int, default=5)
    parser.add_argument('--loss', help="Loss Function (CrossEntropy / NLL)", type=str, default='CrossEntropy')
    parser.add_argument('--optimizer', help="Optimizer (Adam / Adadelta)", type=str, default='Adam')
    parser.add_argument('--lr', help="Learning rate (default: 0.0001)", type=float, default=0.00005)
    parser.add_argument('--weight_decay', help="Weight Decay", type=float, default=0.0)
    parser.add_argument('--momentum', help="Momentum (used for optimizer=SGD)", type=float, default=0.9)
    parser.add_argument('--report_every', help="Report every x number of batches", type=int, default=50)
    parser.add_argument('--cuda_device', help="CUDA Device ID", type=int, default=0)
    parser.add_argument('--batch_size', help="Batch Size", type=int, default=1024)
    parser.add_argument('--target_metric', help="Target Metric to report", type=str, default='micro_average_f1')
    parser.add_argument('--version', help="Run # (0..4)", type=int, default=0)
    parser.add_argument('--memory_loss', help="Loss Function for the memory regularization term", type=str, default='SmoothCrossEntropy')
    parser.add_argument('--memory_reg', help="Memory regularization (not forget the previous model)", type=float, default=0.0)
    parser.add_argument('--teacher_memory', help="Teacher memory (not forget the initial teacher model)", type=float, default=0.0)
    parser.add_argument('--scheduler_gamma', help="Scheduler's multiplier of lr in each epoch", type=float, default=0.1)
    parser.add_argument('--bootstrap_epoch', help="Epoch at which we start the teacher updates", type=int, default=0)
    parser.add_argument('--disable_gpu', help="Disable GPU", action='store_true')

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
    parser.add_argument('--bert_model', help="Type of BERT model: base/large", type=str, default='base')
    parser.add_argument('--simple_aspects', help="Use fine/coarse grained aspects (-1: original A#B label, 0: first part, 1: second part of A#B label", type=int, default=-1)


    # Model Specific
    parser.add_argument('--pretrained_model', help="Pre-trained model", type=str, default='')
    parser.add_argument('--attention', help="Use word attention", action='store_true')
    parser.add_argument('--fix_w_emb', help="Fix word embeddings", action='store_true')
    parser.add_argument('--fix_a_emb', help="Fix aspect embeddings", action='store_true')
    parser.add_argument('--model_type', help="Model type (embedding_based vs bow_based)", type=str, default='embedding_based')
    parser.add_argument('--deep_aspect_clf', help="Use a deep CLF on top of word embeddings", type=str, default='NO')
    parser.add_argument('--teacher_type', help="Teacher Type (v1..3)", type=str, default='v1')
    parser.add_argument('--pos_mass', help="Probability mass to cut from the given aspect and distribute to the remaining aspects", type=float, default=0.2)
    parser.add_argument('--soft_updates', help="Soft (instead of hard) teacher (precision-based) updates (only for v1)", action='store_true')
    parser.add_argument('--hard_teacher_pred', help="Hard aspect predictions per seed word (only the most probable aspect)", action='store_true')
    parser.add_argument('--fix_teacher', help="Fix teacher throughout training (instead of updating)", action='store_true')

    args = parser.parse_args()
    args.enable_gpu = not args.disable_gpu

    seeds = [20, 7, 1993, 42, 127]
    args.seed = seeds[args.version]
    torch.cuda.manual_seed(args.seed)
    seed(args.seed)
    args.num_epochs += args.bootstrap_epoch

    if args.logdir == '../experiments/':
        args.logdir += datetime.now().strftime('%b%d_%H-%M-%S') + '_'

    if args.debug:
        args.logdir = './debug'
        if os.path.exists(args.logdir):
            os.system('rm -rf {}'.format(args.logdir))
    else:
        args.logdir = args.logdir + \
                      "COTRAINING" + \
                      "_att{}".format(args.attention) + \
                      "_fixw{}".format(args.fix_w_emb) + \
                      "_fixa{}".format(args.fix_a_emb) + \
                      "_{}".format(args.loss) + \
                      "_lr{}".format(args.lr) + \
                      "_dropout{}".format(args.emb_dropout) + \
                      '_memloss{}'.format(args.memory_loss) + \
                      '_memreg{}'.format(args.memory_reg) + \
                      "_teacher{}".format(args.teacher_type) + \
                      "_tmem{}".format(args.teacher_memory) + \
                      '_schedgamma{}'.format(args.scheduler_gamma) + \
                      "_bepoch{}".format(args.bootstrap_epoch)

    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    original_logdir = args.logdir
    args.logdir += '/v{}'.format(args.version)
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    args.pretrained_model += '/v{}'.format(args.version)


    print('\t\tEXPERIMENT with domain={}\nargs: {}\nlogdir: {}'.format(args.domain, args, args.logdir))
    run_cotrain(args, args.domain)
