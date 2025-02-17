import time
import datetime
import timeit
import re
import multiprocessing

import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
from nltk import PorterStemmer
from nltk.corpus import stopwords
from ordered_set import OrderedSet

from numpy import array
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, dok_matrix, lil_matrix
from sklearn.feature_extraction.text import CountVectorizer

from src.handOn2 import m1

A = array([[1, 2, 0, 0, 0, 1], [0, 0, 1, 0, 0, 3], [4, 0, 0, 1, 0, 5]])
print(A)

S = coo_matrix(A)
print(S)

print(S.tocsr()[:, 3])

B = S.todense()
print(B)

times = 100000
timeit.timeit(lambda: dok_matrix(B), number=times) / times
timeit.timeit(lambda: lil_matrix(B), number=times) / times
timeit.timeit(lambda: csr_matrix(B), number=times) / times
timeit.timeit(lambda: csc_matrix(B), number=times) / times


def m2_elapsed_experiment_set_specify_input(row: int):
    start = time.time()
    cleaned_description = m1.get_and_clean_data()[:row]
    cleaned_description = cleaned_description.apply(lambda s: re.sub(r'[^A-Za-z]', ' ', s))
    cleaned_description = cleaned_description.apply(lambda s: re.sub(r'\s+', ' ', s))
    # tokenize
    tokenized_description = cleaned_description.apply(lambda s: word_tokenize(s))
    # remove stopwords
    stop_dict = list(stopwords.words())
    sw_removed_description = tokenized_description.apply(lambda s: [word for word in s if word not in stop_dict])
    sw_removed_description = sw_removed_description.apply(lambda s: [word for word in s if len(word) > 2])
    # create stem cache
    concated = np.unique(np.concatenate([s for s in tokenized_description.values]))
    stem_cache = {}
    ps = PorterStemmer()
    for s in concated:
        stem_cache[s] = ps.stem(s)


    stemmed_description = sw_removed_description.apply(lambda s: [ps.stem(word) for word in s])
    end = time.time()
    result_time = end - start
    # print time as min and second format
    result_time_as_float_second = time.strftime("%M:%S", time.gmtime(result_time))
    print(f"Total time stem: {result_time_as_float_second}")


    cv = CountVectorizer(analyzer=lambda x: x)
    X = cv.fit_transform(stemmed_description)
    print(pd.DataFrame(X.toarray(), columns=cv.get_feature_names_out()))
    return stemmed_description, result_time_as_float_second


# replace non alphabeths with spaces, and collapse spaces

stem500, time500 = m2_elapsed_experiment_set_specify_input(500)
stem1000, time1000 = m2_elapsed_experiment_set_specify_input(1000)
stem2000, time2000 = m2_elapsed_experiment_set_specify_input(2000)
