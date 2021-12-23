from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

import threading
import time

from newMaterial import parse_job_description

import newMaterial

cleaned_description = newMaterial.get_and_clean_data()
parsed_description = parse_job_description()


def whole_function():
    sw_set = set(stopwords.words()) - {'c'}
    no_sw_description = parsed_description.apply(lambda x: [w for w in x if w not in sw_set])
    ps = PorterStemmer()
    stemmed_description = no_sw_description.apply(lambda x: set([ps.stem(w) for w in x]))
    all_unique_term = list(set.union(*stemmed_description.to_list()))
    invert_idx = {}
    for s in all_unique_term:
        invert_idx[s] = set(stemmed_description.loc[stemmed_description.apply(lambda x: s in x)].index)
    to_search_db = ['mongodb']
    stemmed_db = np.unique([ps.stem(w) for w in to_search_db])
    searched_db = set.union(*[invert_idx[s] for s in stemmed_db])
    to_search_lang = ['java']
    stemmed_lang = np.unique([ps.stem(w) for w in to_search_lang])
    searched_lang = set.union(*[invert_idx[s] for s in stemmed_lang])
    appear_both = searched_db.intersection(searched_lang)
    print(parsed_description.loc[appear_both].apply(lambda x: ' '.join(x)).tail().to_markdown())


# start = time.perf_counter()
# t = threading.Thread(target=whole_function(), args=[1])
# t.start()
# print(f'Active Threads: {threading.active_count()}')
# t.join()
# end = time.perf_counter()
# print(f'Finished in {round(end - start, 2)} second(s)')
