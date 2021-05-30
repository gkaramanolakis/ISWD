import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import os
from os.path import expanduser
home = expanduser("~")
from sklearn.externals import joblib
import numpy as np
import sys
from bs4 import BeautifulSoup

import sys
import argparse
import re
import os.path
from os import makedirs
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from numpy import log
from scipy.special import rel_entr
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import os
from os.path import expanduser
home = expanduser("~")
from sklearn.externals import joblib
import numpy as np
import sys
sys.path.append(os.path.join(home, "code/research_code/Spring_2018/TextModules/"))
from TextProcessor import TextProcessor



# SemEval Pre-processing scripts
def preprocess_semeval(dataset='english_restaurants', subtask='sentence_level'):
    import semeval_help_functions

    print("\n\n\n\t\t\tDOMAIN={}".format(dataset))
    print('Collecting training data...')
    savefolder = datafolder + "/semeval/preprocessed/"
    method = 'train'
    language = dataset.split('_')[0]
    embeddings_dimension = 300
    pretrained_word_embedding_matrix = os.path.join(datafolder, 'semeval/embeddings/word2emb_{}.pkl'.format(dataset))

    tp = TextProcessor(word_embeddings='custom', embedding_dim=embeddings_dimension,
                       limit=500000, stemming=False, tokenization_method='ruder',
                       tokenizer_language=language,
                       pretrained_word_embedding_matrix_path=pretrained_word_embedding_matrix)
    vocab_size = len(tp.vocab_emb)
    word2id = tp.word2ind_emb
    id2word = {word2id[w]: w for w in word2id}
    joblib.dump(word2id, savefolder + "{}_word2id.pkl".format(dataset))

    # Save word embedding matrix
    print('Collecting word embeddings...')
    word2emb = tp.word2emb
    word_emb_matrix = np.array([word2emb[id2word[i]] for i in range(len(word2id))])
    joblib.dump(word_emb_matrix, savefolder + '{}_word2emb.pkl'.format(dataset))

    review_df, sentence_df, opinion_df = semeval_help_functions.get_reviews_df(dataset, subtask, method)

    aspects = sorted(set(opinion_df['category']))
    aspect2id = {a: i for i, a in enumerate(aspects)}
    joblib.dump(aspects, savefolder + "{}_aspect_names.pkl".format(dataset))

    sentence_df['categories'] = sentence_df['id'].map(
        lambda x: opinion_df[opinion_df['sentence_id'] == x]['category'].tolist())
    sentence_df['label'] = sentence_df['categories'].map(lambda x: [aspect2id[a] for a in x])
    sentence_df['word_ids'] = sentence_df['text'].map(lambda x: tp.get_word_indices(x))
    sentence_df = sentence_df[sentence_df['numofopinions'] > 0]
    joblib.dump(sentence_df, savefolder + "{}_TRAIN.pkl".format(dataset))

    print("Collecting DEV data...")
    test_review_df, test_sentence_df, test_opinion_df = semeval_help_functions.get_reviews_df(dataset, subtask, 'dev')
    test_sentence_df['categories'] = test_sentence_df['id'].map(
        lambda x: test_opinion_df[test_opinion_df['sentence_id'] == x]['category'].tolist())
    test_sentence_df['label'] = test_sentence_df['categories'].map(
        lambda x: [aspect2id[a] if a in aspect2id else -1 for a in x])
    test_sentence_df['word_ids'] = test_sentence_df['text'].map(lambda x: tp.get_word_indices(x))
    test_sentence_df = test_sentence_df[test_sentence_df['numofopinions'] > 0]
    joblib.dump(test_sentence_df, savefolder + "{}_DEV.pkl".format(dataset))

    print("Collecting TEST data...")
    test_review_df, test_sentence_df, test_opinion_df = semeval_help_functions.get_reviews_df(dataset, subtask, 'test')
    test_sentence_df['categories'] = test_sentence_df['id'].map(
        lambda x: test_opinion_df[test_opinion_df['sentence_id'] == x]['category'].tolist())
    test_sentence_df['label'] = test_sentence_df['categories'].map(
        lambda x: [aspect2id[a] if a in aspect2id else -1 for a in x])
    test_sentence_df['word_ids'] = test_sentence_df['text'].map(lambda x: tp.get_word_indices(x))
    test_sentence_df = test_sentence_df[test_sentence_df['numofopinions'] > 0]
    joblib.dump(test_sentence_df, savefolder + "{}_TEST.pkl".format(dataset))
    return


def find_seed_words(dataset='english_restaurants', subtask='sentence_level', method='train', remove_stopwords=True):
    import semeval_help_functions
    datafolder += '/semeval/'
    semeval_help_functions.find_seed_words(dataset=dataset, method=method, subtask=subtask,
                                           savefolder=datafolder + 'seed_words/', remove_stopwords=remove_stopwords)


def preprocess_script():
    all_datasets = ['english_restaurants', 'spanish_restaurants', 'french_restaurants', 'russian_restaurants',
                    'dutch_restaurants', 'turkish_restaurants', 'arabic_hotels', 'english_laptops']
    for dataset in all_datasets:
        preprocess_semeval(dataset)
        find_seed_words(dataset)

        
def read_semeval_file(fpath):
    # Used for reading SemEval 2016 Task 5 Dataset (ABSA)
    xml_data = open(fpath).read()  # Loading the raw XML data
    soup = BeautifulSoup(xml_data, "lxml")
    reviews = soup.find_all('review')
    return reviews


def xml2dict_sentence_level(xml_reviews):
    # Used for reading SemEval 2016 Task 5 Dataset (ABSA)
    restaurant_reviews = []
    for r in xml_reviews:
        review_dict = {}
        review_dict['rid'] = r['rid']
        # print(r['rid'])
        review_dict['text'] = r.getText().strip().replace('\n\n\n\n\n\n', ' ')
        sentences = r.find_all('sentences')
        if len(sentences) > 1:
            print('[WARNING] More than 1 sentences')
        sentences = sentences[0]
        review_dict['sentences'] = []
        for sentence in sentences.find_all('sentence'):
            sentence_dict = {}
            sentence_dict['id'] = sentence['id']
            sentence_dict['text'] = sentence.getText().strip()
            opinions = sentence.find('opinions')
            sentence_dict['opinions'] = []
            if opinions is not None:
                for opinion in opinions.find_all('opinion'):
                    opinion_dict = {}
                    opinion_dict['category'] = opinion['category']
                    opinion_dict['polarity'] = opinion['polarity']
                    try:
                        opinion_dict['from'] = int(opinion['from'])
                        opinion_dict['to'] = int(opinion['to'])
                        opinion_dict['target'] = opinion['target']
                    except:
                        pass
                    sentence_dict['opinions'].append(opinion_dict)
            review_dict['sentences'].append(sentence_dict)
        restaurant_reviews.append(review_dict)
    return restaurant_reviews


def dict2df_sentence_level(review_list):
    # Used for the analysis of SemEval 2016 Task 5 Dataset (ABSA)
    review_df = pd.DataFrame()
    opinions_df = pd.DataFrame()
    sentence_df = pd.DataFrame()
    for review in review_list:
        # print(review)
        review_dict = {}
        review_dict['id'] = review['rid']
        review_dict['text'] = review['text']
        review_dict['numofsentences'] = len(review['sentences'])
        for sentence in review['sentences']:
            sentence_dict = {}
            sentence_dict['id'] = sentence['id']
            sentence_dict['text'] = sentence['text']
            sentence_dict['numofopinions'] = len(sentence['opinions'])

            is_positive = [x for x in sentence['opinions'] if x['polarity'].strip() == 'positive']
            is_negative = [x for x in sentence['opinions'] if x['polarity'].strip() == 'negative']
            if len(is_positive) > 0 and len(is_negative) > 0:
                sentence_dict['has_conflict'] = 1
            else:
                sentence_dict['has_conflict'] = 0
            for opinion in sentence['opinions']:
                # sentence_dict = sentence_dict.update()
                opinion_dict = opinion
                opinion_dict['sentence_id'] = sentence['id']
                opinions_df = opinions_df.append(pd.Series(opinion), ignore_index=True)
            sentence_df = sentence_df.append(pd.Series(sentence_dict), ignore_index=True)
        review_df = review_df.append(pd.Series(review_dict), ignore_index=True)
    return review_df, sentence_df, opinions_df


def get_reviews_df(dataset, subtask, method):
    dataset_path = os.path.join(home,
                                'data1/data/semeval2016_task5/{}/{}/{}_{}.xml'.format(subtask, method, dataset, method))
    if not os.path.exists(dataset_path):
        print("ERROR: dataset not available...")
    print("Loading data for {}: {}".format(dataset, dataset_path))
    reviews = read_semeval_file(dataset_path)
    reviews = xml2dict_sentence_level(reviews)
    review_df, sentence_df, opinions_df = dict2df_sentence_level(reviews)
    return review_df, sentence_df, opinions_df


def find_seed_words(dataset='english_restaurants', method='train', subtask='sentence_level', savefolder='./',
                    remove_stopwords=True, simple_aspects=None):
    # LOAD DATASET
    print("\n\n\n\t\t\tDOMAIN={}".format(dataset))
    print('Loading {} data'.format(method))

    language = dataset.split('_')[0]
    embeddings_dimension = 300
    pretrained_word_embedding_matrix = os.path.join(home, 'data1/data/semeval2016_task5/embedding_matrices/word2emb_{}_protocol2.pkl'.format(dataset))

    tp = TextProcessor(word_embeddings='custom', embedding_dim=embeddings_dimension,
                       limit=500000, stemming=False, tokenization_method='ruder',
                       tokenizer_language=language,
                       pretrained_word_embedding_matrix_path=pretrained_word_embedding_matrix)
    vocab_size = len(tp.vocab_emb)

    review_df, sentence_df, opinion_df = get_reviews_df(dataset, subtask, method)

    if simple_aspects is None:
        aspects = sorted(set(opinion_df['category']))
        aspect2id = {a: i for i, a in enumerate(aspects)}
        sentence_df['categories'] = sentence_df['id'].map(lambda x: opinion_df[opinion_df['sentence_id'] == x]['category'].tolist())
        sentence_df['label'] = sentence_df['categories'].map(lambda x: [aspect2id[a] for a in x])
    elif simple_aspects == 0:
        aspects = sorted(set(opinion_df['category']))
        aspects = sorted(set([a.split('#')[0] for a in aspects]))
        aspect2id = {a: i for i, a in enumerate(aspects)}
        sentence_df['categories'] = sentence_df['id'].map(lambda x: opinion_df[opinion_df['sentence_id'] == x]['category'].tolist())
        sentence_df['categories'] = sentence_df['categories'].map(lambda x: list(set([y.split('#')[0] for y in x])))
        sentence_df['label'] = sentence_df['categories'].map(lambda x: [aspect2id[a] for a in x])
    elif simple_aspects == 1:
        aspects = sorted(set(opinion_df['category']))
        aspects = sorted(set([a.split('#')[1] for a in aspects]))
        aspect2id = {a: i for i, a in enumerate(aspects)}
        sentence_df['categories'] = sentence_df['id'].map(lambda x: opinion_df[opinion_df['sentence_id'] == x]['category'].tolist())
        sentence_df['categories'] = sentence_df['categories'].map(lambda x: list(set([y.split('#')[1] for y in x])))
        sentence_df['label'] = sentence_df['categories'].map(lambda x: [aspect2id[a] for a in x])


    # sentence_df['word_ids'] = sentence_df['text'].map(lambda x: tp.get_word_indices(x))
    sentence_df = sentence_df[sentence_df['numofopinions'] > 0]

    # EXTRACT SEED WORDS USING THE CLARITY SCORING FUNCTION (see Angelidis & Lapata, 2019)
    df = sentence_df
    df['preprocessed_text'] = df['text'].map(lambda x: " ".join(tp.get_word_indices_and_text(x)[0]))
    all_segs = df['preprocessed_text'].tolist()

    aspect_segments = {i: [] for i in range(len(aspects))}

    for ir, row in df.iterrows():
        for aspect_id in row['label']:
            aspect_segments[aspect_id].append(row['preprocessed_text'])

    # compute tfidf scores
    stop_words = stopwords.words(language) if remove_stopwords else None
    vectorizer = TfidfVectorizer(stop_words=stop_words, norm='l1', use_idf=True)
    vectorizer.fit(all_segs)
    gl_freq = vectorizer.transform([' '.join(all_segs)]).toarray()[0]

    # global scores
    gl_scores = {}
    for term, idx in vectorizer.vocabulary_.items():
        gl_scores[term] = gl_freq[idx]

    asp_scores = dict([(aspect, {}) for aspect in aspect_segments.keys()])
    for aspect, segments in aspect_segments.items():

        # aspect-specific scores
        asp_freq = vectorizer.transform([' '.join(segments)]).toarray()[0]

        # entropies correspond to clarity scores
        entropies = rel_entr(asp_freq, gl_freq) / log(2)
        for term, idx in vectorizer.vocabulary_.items():
            asp_scores[aspect][term] = entropies[idx]

        # sort by score and write to file if > 0
        scores = sorted(asp_scores[aspect].items(), reverse=True, key=lambda x: x[1])

        if not os.path.exists(savefolder):
            os.mkdir(savefolder)
        dataset_savefolder = "{}/{}".format(savefolder, dataset)
        if not os.path.exists(dataset_savefolder):
            os.mkdir(dataset_savefolder)
        savefile = '{}/{}_{}.clarity.txt'.format(dataset_savefolder, dataset, aspects[aspect])
        fout = open(savefile, 'w')
        print('Saving seed words to {}'.format(savefile))
        for term, cla in scores[:50]:
            if cla > 0:
                try:
                    fout.write('{0:.5f} {1}\n'.format(cla, term.encode('utf8')))
                except:
                    print("[ERROR] Could not write non-ascii character")
                    # import pdb; pdb.set_trace()
        pkl_savefile = '{}/{}_{}.clarity.pkl'.format(dataset_savefolder, dataset, aspects[aspect])
        joblib.dump(scores, pkl_savefile)

        fout.close()
    joblib.dump(asp_scores, "{}/{}.pkl".format(dataset_savefolder, dataset))
    return
