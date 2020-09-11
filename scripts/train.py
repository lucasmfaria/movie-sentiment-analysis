import numpy as np
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
from joblib import dump
from sklearn.pipeline import Pipeline

pos_train_raw_folder = (Path('..') / 'data' / 'aclImdb_v1/aclImdb/train/pos').resolve()
neg_train_raw_folder = (Path('..') / 'data' / 'aclImdb_v1/aclImdb/train/neg').resolve()
pos_test_raw_folder = (Path('..') / 'data' / 'aclImdb_v1/aclImdb/test/pos').resolve()
neg_test_raw_folder = (Path('..') / 'data' / 'aclImdb_v1/aclImdb/test/neg').resolve()

#Read train data:
pos_train_raw = list()
for file in pos_train_raw_folder.iterdir():
    with open(file, 'r', encoding='utf-8') as f:
        pos_train_raw.append(f.read())
neg_train_raw = list()
for file in neg_train_raw_folder.iterdir():
    with open(file, 'r', encoding='utf-8') as f:
        neg_train_raw.append(f.read())

#Read test data:
pos_test_raw = list()
for file in pos_test_raw_folder.iterdir():
    with open(file, 'r', encoding='utf-8') as f:
        pos_test_raw.append(f.read())

neg_test_raw = list()
for file in neg_test_raw_folder.iterdir():
    with open(file, 'r', encoding='utf-8') as f:
        neg_test_raw.append(f.read())

word_occur = False
use_idf = True
min_df=0.003
max_df=0.98
ngram_range = (1, 2)

pipe = Pipeline([
    ('countvec', CountVectorizer(stop_words=nltk.corpus.stopwords.words('english'), binary=word_occur, min_df=min_df, max_df=max_df, ngram_range=ngram_range)),
    ('tfidf', TfidfTransformer(use_idf=use_idf)),
    ('naivebayes', MultinomialNB())
])

#Prepare train labels:
n_pos_samples_train = len(pos_train_raw)
n_neg_samples_train = len(neg_train_raw)
y_train = np.concatenate((np.ones((n_pos_samples_train)), np.zeros((n_neg_samples_train))))

pipe.fit(pos_train_raw + neg_train_raw, y_train)

file_name = 'pipeline_naivebayes.joblib'
path = Path('../models')
dump(pipe, path / file_name)