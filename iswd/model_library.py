import sys
import os
import argparse
import json
import socket
from datetime import datetime
from time import time
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import xavier_uniform
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import random

def smooth_cross_entropy(input, target, size_average=True):
    # source: https://discuss.pytorch.org/t/cross-entropy-with-one-hot-targets/13580/5
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# For an example of Batch Encoder-Decoder for NMT see: https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
# Some parts of this code are based in https://github.com/EdGENetworks/attention-networks-for-classification
def pad_sequences(sequence_list, enable_gpu=False, min_dim=5, cuda_device=0):
    # Receives a list of sequences with various lengths and pads all sequences with zeros to get the max length.
    # min_dim=5, because when the the padded sequence is given to a CNN with kernel size = 5,... then the input sequence
    # must have at least length=5
    if enable_gpu:
        seq_lengths = torch.LongTensor([len(x) for x in sequence_list]).cuda(cuda_device)
    else:
        # seq_lengths = torch.LongTensor(map(len, sequence_list))
        seq_lengths = torch.LongTensor([len(x) for x in sequence_list])
    max_len = max(seq_lengths.max(), min_dim)
    seq_tensor = autograd.Variable(torch.zeros((len(sequence_list), max_len))).long()
    if enable_gpu:
        seq_tensor = seq_tensor.cuda(cuda_device)
    for idx, (seq, seqlen) in enumerate(zip(sequence_list, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
    if enable_gpu:
        seq_tensor = seq_tensor.cuda(cuda_device)
    return seq_tensor, seq_lengths


class CNN_encoder(nn.Module):
    # Implementation of CNNs for Text Classification based on the paper: Kim, Yoon. "Convolutional neural networks for
    # sentence classification." arXiv preprint arXiv:1408.5882 (2014).
    # Source: https://github.com/Shawn1993/cnn-text-classification-pytorch

    def __init__(self, vocab_size=300000, embedding_dim=300, in_channels=1, out_channels=100,
                 kernel_sizes=[2,3], dropout=0.5, enable_gpu=True, cuda_device=0, batch_norm=False):
        super(CNN_encoder, self).__init__()
        self.lookup = nn.Embedding(vocab_size, embedding_dim)

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes  # A list of kernel sizes (think of them like n-grams) applied in parallel to
                                          # the input
        self.in_channels = in_channels

        # Convolutional Layers
        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(self.in_channels, self.out_channels, (ks, self.embedding_dim)) for ks in self.kernel_sizes])

        # Dropout Layer
        self.dropout = nn.Dropout(dropout)

        self.enable_gpu = enable_gpu
        self.cuda_device = cuda_device
        self.batch_norm=batch_norm

        if self.batch_norm:
            self.bn = nn.BatchNorm1d(embedding_dim, affine=False)

    def forward(self, input_ids):               # embedded_input: (N, W, D) where D = embedding_dim
        seg_ids, _ = pad_sequences(input_ids, min_dim=3, enable_gpu=self.enable_gpu, cuda_device=self.cuda_device)
        embedded_input = self.lookup(seg_ids)
        x = embedded_input.unsqueeze(1)           # x: (N, 1, W, D)
        y = [F.relu(conv(x)).squeeze(3) for conv in self.conv_layers]  # [(N, Co, W), ...] * len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in y]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)  # N * len(Ks)
        if self.batch_norm:
            x = self.bn(x)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        return x


class Attention(nn.Module):
    # type can be 'softmax' or 'sigmoid'
    def __init__(self, input_dim, attention_size=100, attention_type='softmax', bias=False, enable_gpu=True, cuda_device=0):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.attention_size = attention_size
        self.bias = bias
        self.attention_type = attention_type
        self.enable_gpu = enable_gpu
        self.cuda_device=cuda_device
        self.attention_map = nn.Linear(self.input_dim, self.attention_size)
        self.attention_query = nn.Linear(self.attention_size, 1, bias=self.bias)
        return

    def forward(self, input):
        attention_map = F.tanh(self.attention_map(input))  # attention_map: (B, max_num_sentences, sent_GRU_hidden_dim)
        attention_scores = self.attention_query(attention_map)  # attention_scores: (B, max_num_sentences, 1)
        if self.attention_type == 'softmax':
            attention_scores = F.softmax(attention_scores, dim=1)  # attention_scores: (B, max_num_sentences, 1)
            context_vector = torch.bmm(input.transpose(1, 2), attention_scores)  # context_vector: (B, sent_GRU_hidden_dim, 1)
        elif self.attention_type == 'sigmoid':
            attention_scores = F.sigmoid(attention_scores)
            normalized_attention_scores = F.normalize(attention_scores, p=1, dim=1)  # divide attention scores by sum
            context_vector = torch.bmm(input.transpose(1, 2), normalized_attention_scores)  # context_vector: (B, sent_GRU_hidden_dim, 1)
        elif self.attention_type == 'avg':
            # simply return uniform attention weights
            attention_scores = autograd.Variable(torch.ones(input.size(0), input.size(1), 1))
            if self.enable_gpu:
                attention_scores = attention_scores.cuda(self.cuda_device)
            normalized_attention_scores = F.normalize(attention_scores, p=1, dim=1)
            context_vector = torch.bmm(input.transpose(1, 2), normalized_attention_scores)
        else:
            raise(BaseException('Unsupported attention type: {}'.format(self.attention_type)))
        attention_scores = attention_scores.squeeze(2)
        return context_vector, attention_scores


class EmbeddingEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, fix_w_emb=False, pretrained_emb=None):
        super(EmbeddingEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.fix_w_emb = fix_w_emb

        self.encoder = nn.EmbeddingBag(vocab_size, emb_size)

        if pretrained_emb is None:
            xavier_uniform(self.encoder.weight.data)
        else:
            # Initialize with pre-trained word embeddings
            assert pretrained_emb.size() == (vocab_size, emb_size), \
                "Word embedding matrix has incorrect size: {} instead of {}".format(pretrained_emb.size(),
                                                                                    (vocab_size, emb_size))
            self.encoder.weight.data.copy_(pretrained_emb)
        self.encoder.weight.requires_grad = not fix_w_emb

    def forward(self, seg_ids):
        offsets = Variable(torch.arange(0, seg_ids.numel(), seg_ids.size(1), out=seg_ids.data.new().long()))
        enc = self.encoder(seg_ids.view(-1), offsets)
        return enc


class AttentionEncoder(nn.Module):
    """Segment encoder that produces segment vectors as the weighted
    average of word embeddings.
    """

    def __init__(self, vocab_size, emb_size, bias=True, M=None, b=None, pretrained_emb=None, fix_w_emb=False):
        """Initializes the encoder using a [vocab_size x emb_size] embedding
        matrix. The encoder learns a matrix M, which may be initialized
        explicitely or randomly.

        Parameters:
            vocab_size (int): the vocabulary size
            emb_size (int): dimensionality of embeddings
            bias (bool): whether or not to use a bias vector
            M (matrix): the attention matrix (None for random)
            b (vector): the attention bias vector (None for random)
        """
        super(AttentionEncoder, self).__init__()

        self.lookup = nn.Embedding(vocab_size, emb_size)
        if pretrained_emb is None:
            xavier_uniform(self.lookup.weight.data)
        else:
            assert pretrained_emb.size() == (vocab_size, emb_size), \
                "Word embedding matrix has incorrect size: {} instead of {}".format(w_emb.size(), (vocab_size, emb_size))
            self.lookup.weight.data.copy_(pretrained_emb)
        self.lookup.weight.requires_grad = not fix_w_emb

        self.M = nn.Parameter(torch.Tensor(emb_size, emb_size))
        if M is None:
            xavier_uniform(self.M.data)
        else:
            self.M.data.copy_(M)
        if bias:
            self.b = nn.Parameter(torch.Tensor(1))
            if b is None:
                self.b.data.zero_()
            else:
                self.b.data.copy_(b)
        else:
            self.b = None

    def forward(self, inputs):
        """Forwards an input batch through the encoder"""
        x_wrd = self.lookup(inputs)

        # bilinear attention
        x_avg = x_wrd.mean(dim=1)
        x = x_wrd.matmul(self.M)
        x = x.matmul(x_avg.unsqueeze(1).transpose(1, 2))
        if self.b is not None:
            x += self.b

        x = F.tanh(x)  
        a = F.softmax(x, dim=1)

        z = a.transpose(1, 2).matmul(x_wrd)
        z = z.squeeze()
        if z.dim() == 1:
            return z.unsqueeze(0)
        return z


class SeedCLF:
    def __init__(self, id2word, aspects_ids, seed_weights=None, verbose=0, general_ind=4, general_thres=0,
                 hard_pred=False):
        self.id2word = id2word
        self.aspects_ids = aspects_ids
        self.num_aspects = len(aspects_ids)
        self.seed_weights = seed_weights
        self.verbose = verbose
        self.general_ind = general_ind
        self.general_thres = general_thres

        self.seed_dict = self.create_seed_dict()
        self.seed_list = [sw for sw in self.seed_dict]
        self.seed_word_list = [id2word[sw] for sw in self.seed_list]
        self.conf_mat = None
        self.prior = None
        self.hard_pred=hard_pred

    def create_seed_dict(self):
        # Dict: seed_word: (aspect_id, seed_weight)
        seed_dict = {}
        for i, aspect_seeds in enumerate(self.aspects_ids):
            for j, word_id in enumerate(aspect_seeds):
                if self.seed_weights is None:
                    seed_dict[word_id] = (i, 1)
                else:
                    seed_dict[word_id] = (i, self.seed_weights[i][j])
        return seed_dict

    def predict(self, seg):
        seg = list(seg)
        aspect_probs = [0] * self.num_aspects
        for word_id in seg:
            if word_id in self.seed_dict:
                aspect_id, seed_weight = self.seed_dict[word_id]
                aspect_probs[aspect_id] += seed_weight

        if sum(aspect_probs) <= self.general_thres:
            aspect_probs[self.general_ind] = 1

        # Note: This naive classifier will not convert to probabilities.
        # The trainer takes care of this depending on the way of training.
        return aspect_probs

    def predict_verbose(self, seg):
        seg = list(seg)
        aspect_probs = np.zeros(self.num_aspects) #[0] * 9
        seed_pred_dict = {sw:0 for sw in self.seed_dict}

        for word_id in seg:
            if word_id in self.seed_dict:
                if self.conf_mat is None:
                    # use pre-defined weights or do majority voting
                    aspect_id, seed_weight = self.seed_dict[word_id]
                    aspect_probs[aspect_id] += seed_weight
                    seed_pred_dict[word_id] += seed_weight
                else:
                    #aspect_probs[aspect_id] += seed_weight
                    if self.hard_pred == True:
                        # Hard predictions: zero probability for all other aspects
                        # FIXME
                        aspect_id, seed_weight = self.seed_dict[word_id]
                        seed_word_prob=np.zeros(self.num_aspects)
                        seed_word_prob[aspect_id] = self.conf_mat[word_id][aspect_id]
                        aspect_probs += seed_word_prob
                    else:
                        # Soft predictions: non-zero probability for all other aspects
                        aspect_probs += self.conf_mat[word_id]

                    # Report that the teacher used this word
                    seed_pred_dict[word_id] += 1

        if self.prior is not None:
            aspect_probs += self.prior
        elif sum(aspect_probs) <= self.general_thres:
            aspect_probs[self.general_ind] = 1
        seed_pred_list = [seed_pred_dict[sw] for sw in self.seed_list]
        return aspect_probs, seed_pred_list


class LearnableSeedCLF(nn.Module):
    def __init__(self, id2word, aspects_ids, seed_weights=None, verbose=0, general_ind=4, tune=False,
                 enable_gpu=True, cuda_device=0):
        super(LearnableSeedCLF, self).__init__()
        self.id2word = id2word
        self.aspects_ids = aspects_ids
        self.vocab_size = len(id2word)
        self.num_classes = len(aspects_ids)
        self.enable_gpu = enable_gpu
        self.cuda_device = cuda_device
        self.tune = tune
        self.linear = nn.Linear(self.vocab_size, self.num_classes)

        # Initialize the classifier weights with the seed weights
        if seed_weights is None:
            print('Initializing the Seed BoW classifier with UNIFORM seed weights...')
            self.expanded_seeds = torch.zeros((self.vocab_size, self.num_classes))
            for iii, aaa in enumerate(aspects_ids):
                self.expanded_seeds[aaa, iii] = 1.0
        else:
            print('Initializing the Seed BoW classifier with seed weights...')
            self.expanded_seeds = torch.zeros((self.vocab_size, self.num_classes))
            for iii, aaa in enumerate(aspects_ids):
                self.expanded_seeds[aaa, iii] = seed_weights[iii]

        self.expanded_seeds = F.normalize(self.expanded_seeds, p=1, dim=0)
        aspect_ids = [a for i, x in enumerate(aspects_ids) for a in x if i != general_ind]
        self.expanded_seeds[aspect_ids, general_ind] = -1

        self.linear.weight.data.copy_(self.expanded_seeds.transpose(0, 1))
        self.linear.weight.requires_grad = tune

        bias = torch.zeros(self.num_classes)
        bias[general_ind] = 1
        self.linear.bias.data.copy_(bias)
        self.linear.bias.requires_grad = tune

        self.verbose = verbose
        self.general_ind = general_ind

    def forward(self, inputs):
        # Construct (V-dimensional) BoW embeddings given ids
        seg_ids = inputs['ids'].tolist()
        bow_vec = torch.zeros(len(seg_ids), self.vocab_size)
        for i, seg in enumerate(seg_ids):
            for word_id in seg:
                bow_vec[i, word_id] += 1

        if self.enable_gpu:
            bow_vec = bow_vec.cuda(self.cuda_device)

        # Apply the BoW Classifier
        aspect_probs = F.softmax(self.linear(bow_vec), dim=1)
        return aspect_probs


class StudentBoWCLF(nn.Module):
    # This is a 'clever' BoW Classifier. 'Clever', because we set the weights and bias of the general class
    # such that if no aspect specific words appear, then the segment is classified to the 'General' aspect
    def __init__(self, id2word, aspects_ids, enable_gpu=True, cuda_device=0):
        super(StudentBoWCLF, self).__init__()
        self.id2word = id2word
        self.aspects_ids = aspects_ids
        self.vocab_size = len(id2word)
        self.num_classes = len(aspects_ids)
        self.enable_gpu = enable_gpu
        self.cuda_device = cuda_device
        self.linear = nn.Linear(self.vocab_size, self.num_classes)

    def forward(self, inputs):
        # Construct (V-dimensional) BoW embeddings given ids
        seg_ids = inputs['ids'].tolist()
        bow_vec = torch.zeros(len(seg_ids), self.vocab_size)
        for i, seg in enumerate(seg_ids):
            for word_id in seg:
                bow_vec[i, word_id] += 1

        if self.enable_gpu:
            bow_vec = bow_vec.cuda(self.cuda_device)

        # Apply the BoW Classifier
        aspect_logits = F.log_softmax(self.linear(bow_vec), dim=1)
        return aspect_logits

class AspectCLF(nn.Module):
    def __init__(self, vocab_size=3000, num_aspects=9, emb_size=200, enable_gpu=True, cuda_device=0,
                 fix_w_emb=True, fix_a_emb=True, pretrained_emb=None, seed_encodings=None, seed_weights=None,
                 num_seeds=-1, attention=False, deep_clf=False, emb_dropout=0.0, batch_norm=False, use_bert=False,
                 bert_model='base'):
        super(AspectCLF, self).__init__()
        self.num_aspects = num_aspects
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.enable_gpu = enable_gpu
        self.cuda_device = cuda_device
        self.seed_weights = seed_weights
        self.num_seeds = num_seeds
        self.attention = attention
        self.deep_clf = deep_clf
        self.emb_dropout = emb_dropout
        self.batch_norm = batch_norm
        self.use_bert = use_bert
        self.bert_model = bert_model

        if self.use_bert:
            self.emb_size = 768 if self.bert_model == 'base' else 1024

        if self.use_bert:
            print('Using BERT for segment encoding')
        elif not self.attention:
            self.seg_encoder = EmbeddingEncoder(vocab_size, self.emb_size, pretrained_emb=pretrained_emb, fix_w_emb=fix_w_emb)
        else:
            self.seg_encoder = AttentionEncoder(vocab_size, self.emb_size, pretrained_emb=pretrained_emb, fix_w_emb=fix_w_emb)

        if self.emb_dropout > 0:
            self.dropout_layer = nn.Dropout(p=self.emb_dropout)
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(self.emb_size, affine=False)

        # Aspect Embedding
        if seed_encodings is None:
            pass
        else:
            assert seed_encodings.size()[0] == num_aspects and seed_encodings.size()[
                                                                   -1] == self.emb_size, "Aspect embedding matrix has incorrect size"
            if self.seed_weights is None:
                # Aspect Embedding as the *unweighted* avg of seed word embeddings
                seed_encodings = seed_encodings.mean(dim=1)
            else:
                # Aspect Embedding as the *weighted* avg of seed word embeddings
                seed_encodings = seed_encodings.mul(
                    self.seed_weights.double().view(self.num_aspects, self.num_seeds, 1))
                seed_encodings = seed_encodings.sum(dim=1)

        if self.deep_clf == 'MLP':
            self.hidden_layer1 = nn.Linear(self.emb_size, self.emb_size)
            self.hidden_layer2 = nn.Linear(self.emb_size, self.emb_size)
        elif self.deep_clf == 'CNN':
            self.deep_encoder = CNN_encoder(vocab_size=self.vocab_size, embedding_dim=self.emb_size,
                                        enable_gpu=enable_gpu, batch_norm=self.batch_norm,
                                        cuda_device=self.cuda_device)
        self.aspect_clf = nn.Linear(self.emb_size, num_aspects, bias=True)
        if seed_encodings is not None:
            self.aspect_clf.weight.data.copy_(seed_encodings)
        else:
            xavier_uniform(self.aspect_clf.weight.data)
        self.aspect_clf.weight.requires_grad = not fix_a_emb
        self.softmax = nn.Softmax(dim=1)
        return

    def forward(self, inputs):
        seg_ids = inputs['ids'].tolist()
        if self.deep_clf == 'MLP' and not self.use_bert:
            seg_ids, _ = pad_sequences(seg_ids, min_dim=1, enable_gpu=self.enable_gpu, cuda_device=self.cuda_device)
            enc = self.seg_encoder(seg_ids)
            if self.batch_norm:
                enc = F.relu(self.bn(self.hidden_layer2(F.relu(self.bn(self.hidden_layer1(enc))))))
            else:
                enc = F.relu(self.hidden_layer2(F.relu(self.hidden_layer1(enc))))
        elif self.deep_clf == 'CNN':
            enc = self.deep_encoder(seg_ids)
        elif self.use_bert:
            enc = inputs['bert_embeddings']
            enc = Variable(torch.Tensor((enc)))
            if self.enable_gpu:
                enc = enc.cuda(self.cuda_device)
            if self.deep_clf == "MLP" and self.batch_norm:
                enc = F.relu(self.bn(self.hidden_layer2(F.relu(self.bn(self.hidden_layer1(enc))))))
            elif self.deep_clf == "MLP":
                enc = F.relu(self.hidden_layer2(F.relu(self.hidden_layer1(enc))))
        else:
            seg_ids, _ = pad_sequences(seg_ids, min_dim=1, enable_gpu=self.enable_gpu, cuda_device=self.cuda_device)
            enc = self.seg_encoder(seg_ids)
            if self.batch_norm:
                enc = self.bn(enc)

        if self.emb_dropout > 0.0:
            enc = self.dropout_layer(enc)

        a_logits = self.aspect_clf(enc)
        return a_logits



