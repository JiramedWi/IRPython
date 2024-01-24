import time

import pandas as pd
import string
import requests
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from multiprocessing import Pool
from multiprocessing import cpu_count

def get_and_clean_data():
    data = pd.read_csv('resource/software_developer_united_states_1971_20191023_1.csv')
    description = data['job_description']
    cleaned_description = description.apply(lambda s: s.translate(str.maketrans('', '', string.punctuation + u'\xa0')))
    cleaned_description = cleaned_description.apply(lambda s: s.lower())
    cleaned_description = cleaned_description.apply(
        lambda s: s.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), '')))
    cleaned_description = cleaned_description.drop_duplicates()
    return cleaned_description


def simple_tokenize(data):
    cleaned_description = data.apply(lambda s: [x.strip() for x in s.split()])
    return cleaned_description


def parse_job_description():
    cleaned_description = get_and_clean_data()
    cleaned_description = simple_tokenize(cleaned_description)
    return cleaned_description


if __name__ == 'main':
    cleaned_description = parse_job_description()


def count_python_mysql():
    parsed_description = parse_job_description()
    count_python = parsed_description.apply(lambda s: 'python' in s).sum()
    count_mysql = parsed_description.apply(lambda s: 'mysql' in s).sum()
    print('python: ' + str(count_python) + ' of ' + str(parsed_description.shape[0]))
    print('mysql: ' + str(count_mysql) + ' of ' + str(parsed_description.shape[0]))


def parse_db():
    html_doc = requests.get("https://db-engines.com/en/ranking").content
    soup = BeautifulSoup(html_doc, 'html.parser')
    db_table = soup.find("table", {"class": "dbi"})
    all_db = [''.join(s.find('a').findAll(text=True, recursive=False)).strip() for s in
              db_table.findAll("th", {"class": "pad-l"})]
    all_db = list(dict.fromkeys(all_db))
    db_list = all_db[:10]
    db_list = [s.lower() for s in db_list]
    db_list = [[x.strip() for x in s.split()] for s in db_list]
    return db_list


def get_list_of_db():
    parsed_description = parse_job_description()
    cleaned_db = parse_db()

    raw = [None] * len(cleaned_db)
    for i, db in enumerate(cleaned_db):
        raw[i] = parsed_description.apply(lambda s: np.all([x in s for x in db])).sum()
        print(' '.join(db) + ': ' + str(raw[i]) + ' of ' + str(parsed_description.shape[0]))

        with_python = [None] * len(cleaned_db)
    for i, db in enumerate(cleaned_db):
        with_python[i] = parsed_description.apply(lambda s: np.all([x in s for x in db]) and 'python' in s).sum()
        print(' '.join(db) + ' + python: ' + str(with_python[i]) + ' of ' + str(parsed_description.shape[0]))

    for i, db in enumerate(cleaned_db):
        print(' '.join(db) + ' + python: ' + str(with_python[i]) + ' of ' + str(raw[i]) + ' (' +
              str(np.around(with_python[i] / raw[i] * 100, 2)) + '%)')



def inverse_indexing(parsed_description):
    print("in inverse_indexing function")
    sw_set = set(stopwords.words('english')) - {'c'}
    print("start to remove stop words")
    no_sw_description = parsed_description.apply(lambda x: [w for w in x if w not in sw_set])
    ps = PorterStemmer()
    print("start to stem")
    stemmed_description = no_sw_description.apply(lambda x: set([ps.stem(w) for w in x]))
    print("start to get all unique term")
    all_unique_term = list(set.union(*stemmed_description.to_list()))
    invert_idx = {}
    print("start to build invert index")
    for s in all_unique_term:
        print("loop for " + s + " term at " + str(all_unique_term.index(s)) + " of " + str(len(all_unique_term)))
        invert_idx[s] = set(stemmed_description.loc[stemmed_description.apply(lambda x: s in x)].index)
    return invert_idx


def search(invert_idx, query):
    print("in searching function")
    ps = PorterStemmer()
    print("start to stem query")
    processed_query = [s.lower() for s in query.split()]
    stemmed = [ps.stem(s) for s in processed_query]
    print("start to search")
    matched = list(set.intersection(*[invert_idx[s] for s in stemmed]))
    return matched


start_time = time.time()
start_time_gmt = time.gmtime(start_time)
print(time.strftime("%Y-%m-%d %H:%M:%S", start_time_gmt))

parsed_description_new = parse_job_description()
invert_idx = inverse_indexing(parsed_description_new)
query = 'java oracle'
matched = search(invert_idx, query)

end_time = time.time()
result_time = end_time - start_time
result_time_gmt = time.gmtime(result_time)
result_time = time.strftime("%H:%M:%S", result_time_gmt)
print(f"Total time: {result_time}")

print(parsed_description_new.loc[matched].apply(lambda x: ' '.join(x)).head().to_markdown())
