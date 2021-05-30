import h5py
import numpy as np
import sys
import os
import torch
import random
from numpy.random import permutation, seed
from sklearn.externals import joblib

# DataHandler is responsible for data loading and transform operations across all datasets

oposum_domains = ['bags_and_cases', 'keyboards', 'boots', 'bluetooth', 'tv', 'vacuums']
semeval_domains = ['english_restaurants', 'spanish_restaurants', 'french_restaurants', 'russian_restaurants',
                   'dutch_restaurants', 'turkish_restaurants', 'arabic_hotels', 'english_laptops']

# Set data path
datafolder = "../data/"
if not os.path.exists(datafolder):
    raise(BaseException('datafolder not available: {}\nPlease download data first...'.format(os.path.abspath(datafolder))))

def load_data_oposum(domain='bags_and_cases', num_seeds = 30):
    data_folder = datafolder + "oposum/preprocessed/{}".format(domain.upper())

    # Load aspect info
    aspect_file=data_folder+"_aspect_names.txt"
    with open(aspect_file, 'r') as f:
        for line in f:
            aspect_names = line.strip().split('|')
            break
    aspect2id = {a:i for i,a in enumerate(aspect_names)}

    # Load word2id
    fvoc = open(data_folder + '_word_mapping.txt', 'r')
    word2id={}
    id2word={}
    for line in fvoc:
        word, id = line.split()
        id2word[int(id)] = word
        word2id[word] = int(id)
    fvoc.close()

    # Load aspect seeds
    aspect_seeds = datafolder + "oposum/seed_words/{}.{}-weights.txt".format(domain, num_seeds)
    fseed = open(aspect_seeds, 'r')
    aspects_ids = []
    seed_weights = []


    for line in fseed:
        seeds = []
        weights = []
        for tok in line.split():
            word, weight = tok.split(':')
            if word in word2id:
                seeds.append(word2id[word])
                weights.append(float(weight))
            else:
                pass
        aspects_ids.append(seeds)
        seed_weights.append(weights)

    # normalize seed weights to sum to 1 for each aspect
    seed_weights = [list(np.array(s)/np.sum(np.array(s))) for s in seed_weights]
    return word2id, id2word, aspects_ids, seed_weights, aspect_names, aspect2id

def load_data_semeval(domain='english_restaurants', num_seeds = 30):
    # Load aspect info
    data_folder = datafolder + '/semeval/'
    aspect_names = joblib.load(data_folder + "preprocessed/{}_aspect_names.pkl".format(domain))
    seeds_and_weights = joblib.load(data_folder + 'seed_words/{}/{}.pkl'.format(domain, domain))
    aspect2id = {a:i for i,a in enumerate(aspect_names)}

    word2id = joblib.load(data_folder+"preprocessed/{}_word2id.pkl".format(domain))
    id2word = {word2id[w]:w for w in word2id}

    # Load aspect seeds
    aspects_ids = []
    seed_weights = []
    for i in range(len(aspect_names)):
        top = sorted(seeds_and_weights[i].items(), reverse=True, key=lambda x: x[1])
        top = [t for t in top if t[0] in word2id]
        top = top[:num_seeds]
        aspects_ids.append([word2id[t[0]] for t in top])
        seed_weights.append([t[1] for t in top])

    # normalize seed weights to sum to 1 for each aspect
    seed_weights = [list(np.array(s)/np.sum(np.array(s))) for s in seed_weights]
    return word2id, id2word, aspects_ids, seed_weights, aspect_names, aspect2id


def load_data(domain='bags_and_cases', num_seeds=30):
    if domain in oposum_domains:
        return load_data_oposum(domain, num_seeds)
    elif domain in semeval_domains:
        return load_data_semeval(domain, num_seeds)
    else:
        raise(BaseException("domain not supported: {}".format(domain)))


# load test segments
def load_test_segments_oposum(test_file):
    test_batches = []
    test_labels = []
    test_original = []
    test_scodes = []
    f = h5py.File(test_file, 'r')
    for b in f['data']:
        test_batches.extend((f['data/' + b][()]))
        test_labels.extend((f['labels/' + b][()]))
        test_original.extend(list(f['original/' + b][()]))
        test_scodes.extend(list(f['scodes/' + b][()]))
    f.close()
    return test_batches, test_labels, test_original, test_scodes


def load_test_segments_semeval(fpath):
    df = joblib.load(fpath)
    test_segs = df['word_ids'].tolist()
    test_labels = df['label'].tolist()
    test_original = df['text'].tolist()
    test_scodes = df['id'].tolist()

    test_segs_corrected = []
    test_labels_corrected = []
    test_original_corrected = []
    test_scodes_corrected = []

    for i,l in enumerate(test_labels):
        for x in l:
            if x == -1:
                # Found an aspect not in training set...
                continue
            test_labels_corrected.append(x)
            test_segs_corrected.append(test_segs[i])
            test_original_corrected.append(test_original[i])
            test_scodes_corrected.append(test_scodes[i])
    
    # Convert to one-hot
    test_labels_corrected = np.eye(np.max(test_labels_corrected)+1)[test_labels_corrected]
    return test_segs_corrected, test_labels_corrected, test_original_corrected, test_scodes_corrected


def load_wordemb_oposum(fpath):
    id2word = {}
    word2id = {}
    fvoc = open(fpath + '_word_mapping.txt', 'r')
    for line in fvoc:
        word, id = line.split()
        id2word[int(id)] = word
        word2id[word] = int(id)
    fvoc.close()

    f = h5py.File(fpath + '.hdf5', 'r')
    w_emb_array = f['w2v'][()]
    f.close()
    return word2id, w_emb_array

def load_wordemb_semeval(domain):
    word2id = joblib.load(datafolder+'semeval/preprocessed/{}_word2id.pkl'.format(domain))
    w_emb_array = joblib.load(datafolder+'semeval/preprocessed/{}_word2emb.pkl'.format(domain))
    return word2id, w_emb_array

def load_wordemb(domain):
    if domain in oposum_domains:
        w_emb_path = datafolder + "oposum/preprocessed/{}".format(domain.upper())
        return load_wordemb_oposum(w_emb_path)
    elif domain in semeval_domains:
        return load_wordemb_semeval(domain)
    else:
        raise(BaseException("wordemb matrix not available for: {}".format(domain)))


def get_aspect_seeds_oposum(aspect_seeds, word2id):
    fseed = open(aspect_seeds, 'r')
    aspects_ids = []
    seed_weights = []

    for line in fseed:
        seeds = []
        weights = []
        for tok in line.split():
            word, weight = tok.split(':')
            if word in word2id:
                seeds.append(word2id[word])
                weights.append(float(weight))
            else:
                seeds.append(0)
                weights.append(0.0)
        aspects_ids.append(seeds)
        seed_weights.append(weights)
    fseed.close()

    if seed_weights is not None:
        seed_w = torch.Tensor(seed_weights)
        seed_w /= seed_w.norm(p=1, dim=1, keepdim=True)
    else:
        seed_w = None

    return seeds, seed_w, aspects_ids

def get_aspect_seeds_semeval(domain, num_seeds, word2id):
    seed_dict = joblib.load(datafolder + 'semeval/seed_words/{}/{}.pkl'.format(domain, domain))
    aspects_ids = []
    seed_weights = []

    for aspect, seed_word_weights in seed_dict.items():
        scores = sorted(seed_word_weights.items(), reverse=True, key=lambda x: x[1])
        scores = [s for s in scores if s[0] in word2id]
        top_scores = scores[:num_seeds]
        assert len(top_scores) == num_seeds, "too few seed words..."
        aspects_ids.append([word2id[x[0]] for x in top_scores])
        seed_weights.append([float(x[1]) for x in top_scores])

    seed_w = torch.Tensor(seed_weights)
    seed_w /= seed_w.norm(p=1, dim=1, keepdim=True)
    return -1, seed_w, aspects_ids

def get_aspect_embeddings_oposum(aspect_seeds, w_emb_array, word2id):
    # seed initialization (MATE)
    _, seed_w, aspects_ids = get_aspect_seeds_oposum(aspect_seeds, word2id)
    clouds = []
    for seeds in aspects_ids:
        clouds.append(w_emb_array[seeds])
    a_emb = torch.from_numpy(np.array(clouds))
    return a_emb, seed_w

def get_aspect_embeddings_semeval(domain, w_emb_array, num_seeds, word2id):
    _, seed_w, aspects_ids = get_aspect_seeds_semeval(domain, num_seeds, word2id)
    clouds = []
    for seeds in aspects_ids:
        clouds.append(w_emb_array[seeds])
    a_emb = torch.from_numpy(np.array(clouds))
    return a_emb, seed_w


def get_aspect_embeddings(domain, num_seeds, w_emb_array, word2id):
    if domain in oposum_domains:
        aspect_seeds = datafolder + "oposum/seed_words/{}.{}-weights.txt".format(domain, num_seeds)
        return get_aspect_embeddings_oposum(aspect_seeds, w_emb_array, word2id)
    elif domain in semeval_domains:
        return get_aspect_embeddings_semeval(domain, w_emb_array, num_seeds, word2id)
    else:
        raise(BaseException('aspect embeddings not available for {}'.format(domain)))


def get_weak_labels(domain, num_seeds, use_seed_weights=True, pretrained_teacher_folder=''):
    word2id, id2word, aspects_ids, seed_weights, aspect_names, aspect2id = load_data(domain, num_seeds)
    if not use_seed_weights:
        seed_weights = None
    train_segs, train_original, train_scodes = load_train_segments(domain)
    from model_library import SeedCLF

    if domain in oposum_domains:
        general_ind = aspect2id['None']
    elif domain in semeval_domains:
        general_ind = aspect2id['RESTAURANT#GENERAL']
    else:
        raise(BaseException('domain not available: {}'.format(domain)))

    if pretrained_teacher_folder == '':
        # First step of co-training - we haven't yet learned any seed weights (but we can use default weights)
        seedCLF = SeedCLF(id2word, aspects_ids, seed_weights, verbose=0, general_ind=general_ind)
        train_labels = [seedCLF.predict(x) for x in train_segs]
    else:
        # Second step of co-training - we have new estimates of the seed weights
        conf_mat = joblib.load(pretrained_teacher_folder + 'conf_mat.pkl')
        prior = joblib.load(pretrained_teacher_folder + 'prior.pkl')
        print("Using the updated teacher weights and priors: {}".format(prior))
        seedCLF = SeedCLF(id2word, aspects_ids, seed_weights, verbose=0, general_ind=general_ind)
        seedCLF.conf_mat = conf_mat
        print("Running the updated teacher on training data to collect weak labels...")
        train_labels, train_labels_per_seed_word = map(list, zip(*[seedCLF.predict_verbose(x) for x in train_segs]))
    return train_segs, train_labels, train_original, train_scodes


def get_labeled_train_data(domain, num_seeds, one_hot=False, use_seed_weights=True, pretrained_teacher_folder=''):
    # Generate Weak Labels using a Seed Word Classifier.
    train_segs, train_labels, train_original, train_scodes = get_weak_labels(domain, num_seeds, use_seed_weights, pretrained_teacher_folder=pretrained_teacher_folder)
    if not one_hot:
        train_labels = np.argmax(train_labels, axis=1)

    seg_lens = [len(x) for x in train_segs]
    sorted_ind = np.argsort(seg_lens)
    train_segs_sorted = [train_segs[i] for i in sorted_ind]
    train_labels_sorted = [train_labels[i] for i in sorted_ind]
    train_original_sorted = [train_original[i] for i in sorted_ind]
    train_scodes_sorted = [train_scodes[i] for i in sorted_ind]

    return train_segs_sorted, train_labels_sorted, train_original_sorted, train_scodes_sorted


def get_pretrained_bert_embeddings(domain, scodes, method='train', bert_model='base'):
    # Load pre-computed BERT embeddings
    if bert_model != 'base':
        raise (BaseException('unknown bert model: {}'.format(bert_model)))

    if domain in oposum_domains:
        bert_savefolder = datafolder + "pretrained_bert/oposum/"
    elif domain in semeval_domains:
        bert_savefolder=datafolder + "pretrained_bert/semeval/"

    print("Loading BERT ({}) {} embeddings from {}".format(bert_model, method, bert_savefolder))
    # BERT embeddings are not ordered in the same way as train segments. Need to sort based on scodes
    bert_scodes = joblib.load(bert_savefolder+"{}_{}_scodes.pkl".format(domain, method))
    suffix = '_{}'.format(method) if method!='train' else ''
    bert_embeddings = joblib.load(bert_savefolder+"{}_bert_embeddings{}.pkl".format(domain, suffix))

    bert_indices_dict = {scode:i for i,scode in enumerate(bert_scodes)}
    bert_indices = []
    for scode in scodes:
        if scode in bert_indices_dict:
            bert_indices.append(bert_indices_dict[scode])
            prev_scode=scode
        else:
            # There are 4 sentences in the dutch dataset for which there are no BERT embeddings.
            bert_indices.append(bert_indices_dict[prev_scode])
    bert_embeddings_sorted = np.take(bert_embeddings, bert_indices, axis=0)
    print("Done.")
    return bert_embeddings_sorted

def get_train_data(domain):
    train_file = datafolder + "oposum/preprocessed/{}.hdf5".format(domain.upper())
    train_segs, train_original, train_scodes = load_train_segments(train_file)

    seg_lens = [len(x) for x in train_segs]
    sorted_ind = np.argsort(seg_lens)
    train_segs_sorted = [train_segs[i] for i in sorted_ind]
    train_original_sorted = [train_original[i] for i in sorted_ind]

    # Save pre-processed segments & labels
    return train_segs_sorted, train_original_sorted


def load_train_segments_oposum(train_file):
    train_batches = []
    train_original = []
    train_scodes = []
    f = h5py.File(train_file, 'r')
    for b in f['data']:
        train_batches.extend((f['data/' + b][()]))
        train_original.extend(list(f['original/' + b][()]))
        train_scodes.extend(list(f['scodes/' + b][()]))
    f.close()
    return train_batches, train_original, train_scodes


def load_train_segments_semeval(fpath):
    df = joblib.load(fpath)
    train_segs = df['word_ids'].tolist()
    train_original = df['text'].tolist()
    train_scodes = df['id'].tolist()
    return train_segs, train_original, train_scodes

def load_train_segments_semeval_with_labels(domain):
    train_file = datafolder + "semeval/preprocessed/{}_TRAIN.pkl".format(domain)
    df = joblib.load(train_file)
    train_segs = df['word_ids'].tolist()
    # note: here we do not use labels for training
    train_labels = df['label'].tolist()
    train_original = df['text'].tolist()
    train_scodes = df['id'].tolist()
    return train_segs, train_labels, train_original, train_scodes

def load_train_segments(domain):
    if domain in oposum_domains:
        train_file = datafolder + "oposum/preprocessed/{}.hdf5".format(domain.upper())
        return load_train_segments_oposum(train_file)
    elif domain in semeval_domains:
        train_file = datafolder + "semeval/preprocessed/{}_TRAIN.pkl".format(domain)
        return load_train_segments_semeval(train_file)
    else:
        raise(BaseException('can not load train segments for {}'.format(domain)))


def load_dev_segments(domain):
    if domain in oposum_domains:
        dev_file = datafolder + "oposum/preprocessed/{}_DEV.hdf5".format(domain.upper())
        dev_segs, dev_labels, dev_original, dev_scodes = load_test_segments_oposum(dev_file)
        return dev_segs, dev_labels, dev_original, dev_scodes
    elif domain in semeval_domains:
        dev_file = datafolder + "semeval/preprocessed/{}_DEV.pkl".format(domain)
        dev_segs, dev_labels, dev_original, dev_scodes = load_test_segments_semeval(dev_file)
        return dev_segs, dev_labels, dev_original, dev_scodes
    else:
        raise(BaseException('domain not available: {}'.format(domain)))

def load_test_segments(domain):
    if domain in oposum_domains:
        test_file = datafolder + "oposum/preprocessed/{}_TEST.hdf5".format(domain.upper())
        test_segs, test_labels, test_original, test_scodes = load_test_segments_oposum(test_file)
        return test_segs, test_labels, test_original, test_scodes
    elif domain in semeval_domains:
        test_file = datafolder + "semeval/preprocessed/{}_TEST.pkl".format(domain)
        test_segs, test_labels, test_original, test_scodes = load_test_segments_semeval(test_file)
        return test_segs, test_labels, test_original, test_scodes
    else:
        raise(BaseException('domain not available: {}'.format(domain)))


class DataHandler:
    def __init__(self, args):
        torch.manual_seed(args.seed)
        seed(args.seed)
        self.domain = args.domain
        self.num_seeds = args.num_seeds
        self.batch_size = args.batch_size
        self.one_hot = args.one_hot
        self.word2id, self.id2word, self.aspects_ids, self.seed_weights, self.aspect_names, self.aspect2id = load_data(self.domain, self.num_seeds)
        self.num_aspects = len(self.aspects_ids)
        self.flat_aspect_ids = set([i for a in self.aspects_ids for i in a])
        self.seed_word_dropout = args.swd

        if self.domain in oposum_domains:
            self.general_ind = self.aspect2id['None']
        elif self.domain in semeval_domains:
            self.general_ind = self.aspect2id['RESTAURANT#GENERAL']
        else:
            raise (BaseException('domain not available: {}'.format(domain)))

        try:
            self.use_bert = args.use_bert
        except:
            self.use_bert = False

        if args.no_seed_weights:
            self.seed_weights = None

        dev_segs, dev_labels, dev_original, dev_scodes = load_dev_segments(self.domain)
        dev_labels = np.argmax(np.array(dev_labels), axis=1)

        test_segs, test_labels, test_original, test_scodes = load_test_segments(self.domain)
        test_labels = np.argmax(np.array(test_labels), axis=1)


        # Get seed word embeddings
        self.word2id, self.w_emb_array = load_wordemb(self.domain)
        self.w_emb = torch.from_numpy(self.w_emb_array)
        self.vocab_size, self.emb_size = self.w_emb.size()

        # Compute aspect matrix
        self.a_emb, self.seed_w = get_aspect_embeddings(self.domain, self.num_seeds, self.w_emb_array, self.word2id)

        # Get segments and weak labels
        print('Loading training data...')
        pretrained_teacher_folder = ''

        train_segs, train_labels, train_original, train_scodes = get_labeled_train_data(self.domain, self.num_seeds, one_hot=self.one_hot,
                                                          use_seed_weights=not args.no_seed_weights,
                                                          pretrained_teacher_folder=pretrained_teacher_folder)


        if self.use_bert:
            self.train_bert_embeddings = get_pretrained_bert_embeddings(self.domain, train_scodes, method='train', bert_model=args.bert_model)
            self.dev_bert_embeddings = get_pretrained_bert_embeddings(self.domain, dev_scodes, method='dev', bert_model=args.bert_model)
            self.test_bert_embeddings = get_pretrained_bert_embeddings(self.domain, test_scodes, method='test', bert_model=args.bert_model)

        print('Done...')

        self.train_segs, self.train_labels, self.train_original, self.train_scodes = train_segs, train_labels, train_original, train_scodes
        self.dev_segs, self.dev_labels = dev_segs, dev_labels
        self.test_segs, self.test_labels = test_segs, test_labels

        self.dev_scodes = dev_scodes
        self.test_scodes = test_scodes

    def get_train_batches(self):
        batch_size = self.batch_size
        total = len(self.train_segs)
        total_batches = int(len(self.train_segs) / batch_size) + 1

        if self.seed_word_dropout > 0.0:
            print('\n DROPPING ASPECT RELATED WORDS FROM TEACHER INPUT with probability {}'.format(self.seed_word_dropout))
            train_segs = [np.array([self.swdrop(ind, self.seed_word_dropout) for ind in seg]) for seg in self.train_segs]
        else:
            train_segs = self.train_segs

        for i, batch_st in enumerate(range(0, total, batch_size)):
            batch_end = min(batch_st + batch_size, total)

            batch_segs = train_segs[batch_st:batch_end]
            batch_teacher_segs = self.train_segs[batch_st:batch_end]
            batch_labels = self.train_labels[batch_st:batch_end]
            batch_original = self.train_original[batch_st:batch_end]
            bert_embeddings = self.train_bert_embeddings[batch_st:batch_end] if self.use_bert else []
            yield {
                'ind': i,
                'ids': np.array(batch_segs),
                'teacher_ids': np.array(batch_teacher_segs),
                'label': np.array(batch_labels),
                'original': np.array(batch_original),
                'bert_embeddings': bert_embeddings,
                'total': total_batches
            }

    def swdrop(self, ind, p):
        # if seed word dropout is 0, then we always keep the seed word
        # if it is >0, then there is swd dropout probability to drop it
        if ind in self.flat_aspect_ids and random.random() < p:
            return 0
        else:
            return ind

    def get_eval_batches(self):
        batch_size = 1
        total = len(self.dev_segs)
        total_batches = int(len(self.dev_segs) / batch_size) + 1
        for i, batch_st in enumerate(range(0, total, batch_size)):
            batch_end = min(batch_st + batch_size, total)

            batch_segs = self.dev_segs[batch_st:batch_end]
            batch_labels = self.dev_labels[batch_st:batch_end]
            batch_scodes = self.dev_scodes[batch_st:batch_end]
            if len(batch_segs) == 0:
                continue

            bert_embeddings = self.dev_bert_embeddings[batch_st:batch_end] if self.use_bert else []


            yield {
                'ind': i,
                'ids': np.array(batch_segs), 
                'label': np.array(batch_labels),
                'bert_embeddings': bert_embeddings,
                'scodes': batch_scodes,
                'total': total_batches
            }

    def get_test_batches(self):
        batch_size = 1
        total = len(self.test_segs)
        total_batches = int(len(self.test_segs) / batch_size) + 1
        for i, batch_st in enumerate(range(0, total, batch_size)):
            batch_end = min(batch_st + batch_size, total)

            batch_segs = self.test_segs[batch_st:batch_end]
            batch_labels = self.test_labels[batch_st:batch_end]
            batch_scodes = self.test_scodes[batch_st:batch_end]
            if len(batch_segs) == 0:
                continue

            bert_embeddings = self.test_bert_embeddings[batch_st:batch_end] if self.use_bert else []

            yield {
                'ind': i,
                'ids': np.array(batch_segs), 
                'label': np.array(batch_labels),
                'bert_embeddings': bert_embeddings,
                'scodes': batch_scodes,
                'total': total_batches
            }