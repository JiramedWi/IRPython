from pathlib import Path

import pandas as pd
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from requests.auth import HTTPBasicAuth
from sklearn.feature_extraction.text import TfidfVectorizer

from src.Webcrawler.bm25 import BM25
from elasticsearch import Elasticsearch


# Read all csv file in crawled folder
def read_crawled_data():
    path = '/crawled'
    all_files = Path(path).rglob('*.csv')
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
    df_result = pd.concat(li, axis=0, ignore_index=True)
    df_result.to_pickle(Path(path) / 'crawled_data_df.pkl')
    return df_result


def preProcess(s):
    ps = PorterStemmer()
    s = word_tokenize(s)
    stopwords_set = set(stopwords.words())
    stop_dict = {s: 1 for s in stopwords_set}
    s = [w for w in s if w not in stop_dict]
    s = [ps.stem(w) for w in s]
    s = ' '.join(s)
    return s


# # data = pd.read_pickle('crawled/crawled_data_df.pkl')
# tfidf_vectorizer = TfidfVectorizer(preprocessor=preProcess, stop_words=stopwords.words('english'))
# bm25 = BM25(tfidf_vectorizer)
# bm25.fit(data.apply(lambda s: ' '.join(s[['title', 'text']]), axis=1))

es = Elasticsearch("https://localhost:9200", basic_auth=("elastic", "wfElebF0u9Z*h+xdgv6+"),
                   ca_certs="/Users/Jumma/http_ca.crt")

es.info().body
