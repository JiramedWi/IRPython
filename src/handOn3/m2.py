import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

import m1

cleaned_description = m1.get_and_clean_data()
cleaned_description = cleaned_description.iloc[:2]

tokenized_description = cleaned_description.apply(lambda s: word_tokenize(s))
sw_removed_description = tokenized_description.apply(lambda s: [word for word in s if word not in stopwords.words()])
sw_removed_description = sw_removed_description.apply(lambda s: [word for word in s if len(word) > 2])

ps = PorterStemmer()
stemmed_description = sw_removed_description.apply(lambda s: [ps.stem(w) for w in s])

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(analyzer=lambda x: x)
X = cv.fit_transform(stemmed_description)

print(pd.DataFrame(X.toarray(), columns=cv.get_feature_names()))
print(pd.DataFrame(X.toarray(), columns=cv.get_feature_names()).to_markdown())
print(pd.DataFrame(X.toarray(), columns=cv.get_feature_names()).T.to_markdown())

# CSR
print(print(X[0, :]))

# COO
print(print(X.tocoo()[0, :]))

import timeit

# Non-compress format
XX = X.toarray()
timeit.timeit(lambda: np.matmul(XX, XX.T), number=1)
# np.shape(np.matmul(X.toarray(), X.toarray().T))

# CSR format
timeit.timeit(lambda: X * X.T, number=1)
# np.shape(X * X.T))
