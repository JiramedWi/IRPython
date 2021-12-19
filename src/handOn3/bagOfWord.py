from turtle import pd

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

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


def sk_vectorize():
    cleaned_description = m1.get_and_clean_data()
    vectorizer = CountVectorizer(preprocessor=preProcess)
    vectorizer.fit_transform(cleaned_description)
    query = vectorizer.transform(['good at java and python'])
    print(query)
    print(vectorizer.inverse_transform(query))

    vectorizer = CountVectorizer(preprocessor=preProcess, ngram_range=(1, 2))
    X = vectorizer.fit_transform(cleaned_description)
    print(vectorizer.get_feature_names())

    N = 5
    cleaned_description = m1.get_and_clean_data()
    cleaned_description = cleaned_description.iloc[:N]
    vectorizer = CountVectorizer(preprocessor=preProcess)
    X = vectorizer.fit_transform(cleaned_description)
    print(X.toarray())

    # idf = N / (X.tocoo()>0).sum(0)
    X.data = np.log10(X.data + 1)
    X.data = X.multiply(np.log10(N / X.sum(0))[0])
    print(X.toarray())
    print(pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names()))
