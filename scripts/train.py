import numpy as np
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
from joblib import dump

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

countvec = CountVectorizer(stop_words=nltk.corpus.stopwords.words('english'), binary=False,
                          min_df=0.003, max_df=0.98, ngram_range=(1, 2)).fit(pos_train_raw + pos_test_raw + neg_train_raw + neg_test_raw)

#Prepare train data:
X_train = countvec.transform(pos_train_raw + pos_test_raw + neg_train_raw + neg_test_raw)
tfidf = TfidfTransformer(use_idf=True).fit(X_train)
X_train = tfidf.transform(X_train)

n_pos_samples_train = len(pos_train_raw + pos_test_raw)
n_neg_samples_train = len(neg_train_raw + neg_test_raw)
y_train = np.concatenate((np.ones((n_pos_samples_train)), np.zeros((n_neg_samples_train))))

#Fit model:
naivebayes = MultinomialNB()
naivebayes.fit(X_train, y_train)

file_name = 'naivebayes.joblib'
path = Path('../models')
dump(naivebayes, path / file_name)