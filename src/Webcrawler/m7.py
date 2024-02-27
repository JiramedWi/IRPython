import pickle
import re
import string
import time

import joblib
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from multiprocessing.pool import ThreadPool as Pool

from sklearn import model_selection, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb

stopwords_set = set(stopwords.words('English'))
ps = PorterStemmer()


def preprocess(text):
    cleaned_text = text.translate(str.maketrans('', '', '!"#$%&\'()*+,.<=>?@[]^`{|}~' + u'\xa0'))
    cleaned_text = cleaned_text.lower()
    cleaned_text = cleaned_text.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), ''))
    cleaned_text = ' '.join(['_variable_with_underscore' if '_' in t else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_variable_with_dash' if '-' in t else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_long_variable_name' if len(t) > 15 and t[0] != '#' else t for t in
                             cleaned_text.split()])
    cleaned_text = ' '.join(['_weburl' if t.startswith('http') and '/' in t else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_number' if re.sub('[\\/;:_-]', '', t).isdigit() else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_variable_with_address' if re.match('.*0x[0-9a-f].*', t) else t for t in
                             cleaned_text.split()])
    cleaned_text = ' '.join(['_name_with_number' if re.match('.*[a-f]*:[0-9]*', t) else t for t in
                             cleaned_text.split()])
    cleaned_text = ' '.join(['_number_starts_with_one_character' if re.match('[a-f][0-9].*', t) else t for t in
                             cleaned_text.split()])
    cleaned_text = ' '.join(['_number_starts_with_three_characters' if re.match('[a-f]{3}[0-9].*', t) else t for t in
                             cleaned_text.split()])
    cleaned_text = ' '.join(['_version' if any(i.isdigit() for i in t) and t.startswith('v') else t for t in
                             cleaned_text.split()])
    cleaned_text = ' '.join(['_localpath' if ('\\' in t or '/' in t) and ':' not in t else t for t in
                             cleaned_text.split()])
    cleaned_text = ' '.join(['_image_size' if t.endswith('px') else t for t in cleaned_text.split()])
    tokenized_text = word_tokenize(cleaned_text)
    sw_removed_text = [word for word in tokenized_text if word not in stopwords_set]
    sw_removed_text = [word for word in sw_removed_text if len(word) > 2]
    stemmed_text = ' '.join([ps.stem(w) for w in sw_removed_text])
    return stemmed_text


stopword_set = None
stemmer = None


def initialize_pool(stopwords_arg, stemmer_arg):
    global stopword_set
    global stemmer
    stopword_set = stopwords_arg
    stemmer = stemmer_arg


# # Start time
# start = time.time()
# print("Start pool things")
# dataset = pd.read_json('resource/embold_train.json')
# dataset.loc[dataset['label'] > 0, 'label'] = -1
# dataset.loc[dataset['label'] == 0, 'label'] = 1
# dataset.loc[dataset['label'] == -1, 'label'] = 0
# stopwords = set(stopwords.words('English'))
# pool = Pool(8, initializer=initialize_pool, initargs=(stopwords, ps))
# cleaned_title = pool.map(preprocess, dataset.title)
# cleaned_body = pool.map(preprocess, dataset.body)
# # End time in minutes
# end = time.time()
# print("Time taken in minutes: ", (end - start) / 60)
# pool.close()

# data_texts = pd.DataFrame([cleaned_title, cleaned_body], index=['title', 'body']).T
# joblib.dump(data_texts, 'resource/data_texts.pkl')
# y = dataset['label']
# joblib.dump(y, 'resource/y.pkl')

data_texts = joblib.load('resource/data_texts.pkl')
y = joblib.load('resource/y.pkl')
data_fit, data_blindtest, y_fit, y_blindtest = model_selection.train_test_split(data_texts, y, test_size=0.1)
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
cleaned_title = data_fit['title']
cleaned_body = data_fit['body']
tfidf_vectorizer.fit(cleaned_title + cleaned_body)
X_tfidf_fit = tfidf_vectorizer.transform(data_fit['title'])

gbm_model = lgb.LGBMClassifier()
precision_cv_score = model_selection.cross_val_score(gbm_model, X_tfidf_fit, y_fit, cv=5, n_jobs=-2,
                                                     scoring='precision_macro').mean()
recall_cv_score = model_selection.cross_val_score(gbm_model, X_tfidf_fit, y_fit, cv=5, n_jobs=-2,
                                                  scoring='recall_macro').mean()
f1_cv_score = model_selection.cross_val_score(gbm_model, X_tfidf_fit, y_fit, cv=5, n_jobs=-2,
                                              scoring='f1_macro').mean()
print('CV_result: Precision:{0:.4f} Recall:{1:.4f} F1_cv_score:{2:.4f}'.format(precision_cv_score, recall_cv_score,
                                                                               f1_cv_score))

data_fit_train, data_fit_test, y_fit_train, y_fit_test = model_selection.train_test_split(data_fit, y_fit,
                                                                                          test_size=0.3)
X_tfidf_fit_train = tfidf_vectorizer.transform(data_fit_train['title'])
X_tfidf_fit_test = tfidf_vectorizer.transform(data_fit_test['title'])
X_tfidf_blindtest = tfidf_vectorizer.transform(data_blindtest['title'])

gbm_model.fit(X_tfidf_fit_train, y_fit_train, eval_set=[(X_tfidf_fit_test, y_fit_test)],
              eval_metric='AUC')
precision_test_score = metrics.precision_score(gbm_model.predict(X_tfidf_blindtest), y_blindtest, average='macro')
recall_test_score = metrics.recall_score(gbm_model.predict(X_tfidf_blindtest), y_blindtest,
                                         average='macro')
f1_test_score = metrics.f1_score(gbm_model.predict(X_tfidf_blindtest), y_blindtest,
                                 average='macro')
print('Test_result: Precision:{0:.4f} Recall:{1:.4f} F1_score:{2:.4f}'.format(precision_test_score, recall_test_score,
                                                                              f1_test_score))
pickle.dump(tfidf_vectorizer, open('resource/github_bug_prediction_tfidf_vectorizer.pkl', 'wb'))
pickle.dump(gbm_model, open('resource/github_bug_prediction_basic_model.pkl', 'wb'))

