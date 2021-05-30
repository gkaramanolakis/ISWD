from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
from copy import deepcopy
import socket
import torch.nn.functional as F
from torch.autograd import Variable
from collections import defaultdict

def flatten_dict(res_dict):
    # This is a help function used to flatten a dict that has a depth larger than 1
    # i.e. replace every list value by multiple scalar values so that it can be processed by tensorboard
    # Note numpy.int64 cannot be stored in a JSON format, so we cast it to int
    flattened_dict = {}
    for x in res_dict:
        try:
            n = len(res_dict[x])
            # x is a list
            for i in range(n):
                try:
                    ll = len(res_dict[x][i])
                    # x is a 2d matrix (list of lists)
                    for j in range(ll):
                        flattened_dict[x + '_' + str(i) + '_' + str(j)] = int(res_dict[x][i][j])
                except:
                    flattened_dict[x + '_' + str(i)] = float(res_dict[x][i])
                    pass
        except TypeError:
            # x is not a list
            flattened_dict[x] = float(res_dict[x])
            pass
    return flattened_dict


class Evaluator:
    def __init__(self, classes=[0, 1], class_labels=['neg', 'pos']):
        self.results = {}
        self.flattened_result_dict = {}  # used for tensorboard visualization
        return

    def evaluate(self, group_preds, group_labels, group_proba=None, verbose=1, gt_classes=[0,1]):
        # Compute evaluation metrics over the group predictions
        res = {}
        y_pred = group_preds
        y_true = group_labels
        y_proba = group_proba

        res['acc'] = metrics.accuracy_score(y_pred=y_pred, y_true=y_true)
        res['prec'], res['rec'], res['f1'], support = metrics.precision_recall_fscore_support(y_pred=y_pred, y_true=y_true)
        res['macro_average_f1'] = np.mean(res['f1'])
        res['micro_average_f1'] = metrics.f1_score(y_pred=y_pred, y_true=y_true, average='micro')
        res['conf_mat'] = metrics.confusion_matrix(y_pred=y_pred, y_true=y_true)

        if y_proba is not None:
            if y_proba.shape[1] != len(gt_classes):
                raise(BaseException("y_proba.shape ({}) != len(gt_classes) ({})".format(y_proba.shape[1], len(gt_classes))))
            if y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
            try:
                y_true_binarized = label_binarize(y_true, classes=gt_classes)
                res['roc-auc'] = metrics.roc_auc_score(y_true=y_true_binarized, y_score=y_proba, average='macro')
            except:
                if verbose:
                    print('[WARNING] Could not compute the roc-auc score...')
                res['roc-auc'] = -1
        else:
            res['roc-auc'] = -1.0
        self.flattened_result_dict = flatten_dict(res)
        self.results = res
        if verbose:
            self.print_report()
        return res

    def print_report(self):
        res = self.results
        print("Accuracy: {:f}".format(res['acc']))
        print("Precision: {}".format(res['prec']))
        print("Recall: {}".format(res['rec']))
        print("F1: {}".format(res['f1']))
        print("AUC: {}".format(res['roc-auc']))
        print("Confusion matrix:\n{}".format(res['conf_mat']))
