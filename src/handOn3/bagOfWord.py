import re

import joblib
import pandas as pd

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from ordered_set import OrderedSet
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import m1
import bm25


def preProcess(s):
    ps = PorterStemmer()
    s = word_tokenize(s)
    stopwords_set = set(stopwords.words())
    stop_dict = {s: 1 for s in stopwords_set}
    s = [w for w in s if w not in stop_dict]
    s = [ps.stem(w) for w in s]
    s = ' '.join(s)
    return s


# def sk_vectorize():
#     cleaned_description = m1.get_and_clean_data()
#     vectorizer = CountVectorizer(preprocessor=preProcess)
#     vectorizer.fit_transform(cleaned_description)
#     query = vectorizer.transform(['good at java and python'])
#     print(query)
#     print(vectorizer.inverse_transform(query))
# vectorizer = CountVectorizer(preprocessor=preProcess, ngram_range=(1, 2))
# X = vectorizer.fit_transform(cleaned_description)
# print(vectorizer.get_feature_names())
#
# N = 5
# cleaned_description = m1.get_and_clean_data()
# cleaned_description = cleaned_description.iloc[:N]
# vectorizer = CountVectorizer(preprocessor=preProcess)
# X = vectorizer.fit_transform(cleaned_description)
# print(X.toarray())
#
# # idf = N / (X.tocoo()>0).sum(0)
# X.data = np.log10(X.data + 1)
# X.data = X.multiply(np.log10(N / X.sum(0))[0])
# print(X.toarray())
# print(pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names()))


def sk_vectorize(texts, vectorizer):
    # my_custom_preprocessor = create_custom_preprocessor(stop_dict, stem_cache)
    # vectorizer = CountVectorizer(preprocessor=my_custom_preprocessor)
    # vectorizer.fit(cleaned_description)
    query = vectorizer.transform(texts)
    print(query)
    print(vectorizer.inverse_transform(query))


def vectorize_preprocess(cleaned_description, stop_dict, stem_cache, ngram_range):
    my_custom_preprocessor = create_custom_preprocessor(stop_dict, stem_cache)
    vectorizer = CountVectorizer(preprocessor=my_custom_preprocessor, ngram_range=ngram_range)
    vectorizer.fit(cleaned_description)
    return vectorizer


def create_stem_cache(cleaned_description):
    tokenized_description = cleaned_description.apply(lambda s: word_tokenize(s))
    concated = np.unique(np.concatenate([s for s in tokenized_description.values]))
    stem_cache = {}
    ps = PorterStemmer()
    for s in concated:
        stem_cache[s] = ps.stem(s)
    return stem_cache


def create_custom_preprocessor(stop_dict, stem_cache):
    def custom_preprocessor(s):
        ps = PorterStemmer()
        s = re.sub(r'[^A-Za-z]', ' ', s)
        s = re.sub(r'\s+', ' ', s)
        s = word_tokenize(s)
        s = list(OrderedSet(s) - stop_dict)
        s = [word for word in s if len(word) > 2]
        s = [stem_cache[w] if w in stem_cache else ps.stem(w) for w in s]
        s = ' '.join(s)
        return s

    return custom_preprocessor


cleaned_description = m1.get_and_clean_data()
stem_cache = create_stem_cache(cleaned_description)
stop_dict = set(stopwords.words('English'))
my_custom_preprocessor = create_custom_preprocessor(stop_dict, stem_cache)
vectorizer = vectorize_preprocess(cleaned_description, stop_dict, stem_cache, (1, 2))
joblib.dump(vectorizer, './resource/vectorizer_n_2.pkl')
print('vectorizer saved')
vectorizer = joblib.load('./resource/vectorizer.pkl')
vectorizer_n_2 = joblib.load('./resource/vectorizer_n_2.pkl')
sk_vectorize(['python is simpler than java'], vectorizer)


tf_idf_vectorizer = TfidfVectorizer(preprocessor=my_custom_preprocessor, use_idf=True)
tf_idf_vectorizer.fit(cleaned_description)
transformed_data = tf_idf_vectorizer.transform(cleaned_description)
X_tfidf_df = pd.DataFrame(transformed_data.toarray(), columns=tf_idf_vectorizer.get_feature_names_out())
max_term = X_tfidf_df.sum().sort_values()[-10:].sort_index().index
print(X_tfidf_df[max_term].head(5).to_markdown())

query = ['product manager who can also provide supports in documentation']
query = ['aws devops']
transformed_query = tf_idf_vectorizer.transform(query)
transformed_query_df = pd.DataFrame(transformed_query.toarray(),
                                    columns=tf_idf_vectorizer.get_feature_names_out()).loc[0]
print('new query')
print(transformed_query_df[max_term].to_markdown())
q_dot_d = X_tfidf_df.dot(transformed_query_df.T)
print(q_dot_d.sort_values(ascending=False).head(5).to_markdown())

print(cleaned_description.iloc[np.argsort(q_dot_d)[::-1][:5].values].to_markdown())
