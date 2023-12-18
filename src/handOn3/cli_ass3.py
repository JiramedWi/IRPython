from turtle import pd

import typer
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from pip._vendor.distlib.compat import raw_input
from sklearn.feature_extraction.text import CountVectorizer

import m1
import bm25

app = typer.Typer()


def preProcess(s):
    ps = PorterStemmer()
    s = word_tokenize(s)
    stopwords_set = set(stopwords.words())
    stop_dict = {s: 1 for s in stopwords_set}
    s = [w for w in s if w not in stop_dict]
    s = [ps.stem(w) for w in s]
    s = ' '.join(s)
    return s


@app.command()
def vtr():
    inputQuery = raw_input("Input your query:")
    inputRange = input("Input range too:")
    inputMethod = raw_input("Type method that you want(tf or tfidf or bm25):")
    cleaned_description = m1.get_and_clean_data()
    vectorizer = CountVectorizer(preprocessor=preProcess, ngram_range=(1, 2))
    vectorizer.fit_transform(cleaned_description)
    query = vectorizer.transform([inputQuery])
    print(query)
    print(vectorizer.inverse_transform(query))
    X = vectorizer.fit_transform(cleaned_description)
    inputRange = int(inputRange)
    cleaned_description.iloc[:inputRange]
    if (inputMethod == "tf"):
        print(X.toarray())
    elif (inputMethod == "tfidf"):
        X.data = np.log10(X.data + 1)
        X.data = X.multiply(np.log10(inputRange / X.sum(0))[0])
        print(X.toarray())
        print(pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names()))
    elif (inputMethod == "bm25"):
        bm25.fit(cleaned_description)
        print(bm25.transform(inputQuery, cleaned_description))
    else:
        print("? where is the method run it again!!")


@app.command()
def bye():
    print("goodbye")


if __name__ == "__main__":
    app()
