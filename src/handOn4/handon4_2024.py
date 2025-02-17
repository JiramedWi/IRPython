import itertools
import os
import functools
import multiprocessing as mp
import re
import time

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from string import ascii_lowercase

topdir = "/Users/Jumma/git_repo/IRPython/src/handOn4/EN"
all_contents = []
for dirpath, dirnames, filenames in os.walk(topdir):
    for filename in filenames:
        if filename.endswith('plain.txt'):
            with open(os.path.join(dirpath, filename), 'r') as file:
                # print(file)
                all_contents.append(file.read())


def preProcess(text):
    # Step 1: Remove all non-English characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Step 2: Replace multiple whitespaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Step 3: Convert all letters to lowercase
    text = text.lower()
    # Step 4: Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


processed_content = [preProcess(s) for s in all_contents]

vectorizer = CountVectorizer()
vectorizer.fit(processed_content)
freq_iula = vectorizer.transform(processed_content)
freq_iula = pd.DataFrame(freq_iula.todense(), columns=vectorizer.get_feature_names_out()).sum()
total = freq_iula.sum()

query = ['deet', 'deft', 'defer', 'defect', 'defeat']
transformed_query = [vectorizer.inverse_transform(vectorizer.transform([q])) for q in query]
query_freq = pd.Series([freq_iula.T.loc[tq[0]].values[0] if len(tq[0]) > 0 else 0 for tq in transformed_query],
                       index=query)

norvig_orig = pd.read_csv('http://norvig.com/ngrams/count_big.txt',
                          sep='\t', encoding="ISO-8859-1", header=None)
norvig_orig = norvig_orig.dropna()
norvig_orig.columns = ['term', 'freq']
norvig_orig.head()

norvig = pd.read_csv('http://norvig.com/ngrams/count_1edit.txt',
                     sep='\t', encoding="ISO-8859-1", header=None)
norvig.columns = ['term', 'edit']
norvig = norvig.set_index('term')
norvig.head()


def get_count(c, norvig_orig):
    return norvig_orig.apply(lambda x: x.term.count(c) * x.freq, axis=1).sum()


character_set = list(map(''.join, itertools.product(ascii_lowercase, repeat=1))) + \
                list(map(''.join, itertools.product(ascii_lowercase, repeat=2)))

#start time to run the code
start = time.time()
with mp.Pool(processes=8) as pool:
    freq_list = pool.map(functools.partial(
        get_count, norvig_orig=norvig_orig), character_set)
end = time.time()
result_time = end - start
# print time as min and second format
result_time_as_float_second = time.strftime("%M:%S", time.gmtime(result_time))
print(f"Total time stem: {result_time_as_float_second}")

freq_df = pd.DataFrame([character_set, freq_list], index=['char', 'freq']).T
freq_df = freq_df.set_index('char')
