import joblib
from sklearn import model_selection
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import lightgbm as lgb

count_vectorizer = CountVectorizer(ngram_range=(1, 2))

data_texts = joblib.load('resource/data_texts.pkl')
data_texts = data_texts[:1000]
y = joblib.load('resource/y.pkl')
y = y[:1000]
data_fit, data_blindtest, y_fit, y_blindtest = model_selection.train_test_split(data_texts, y, test_size=0.1)
cleaned_title = data_fit['title']
cleaned_body = data_fit['body']
count_vectorizer.fit(cleaned_title + cleaned_body)
X_tf_fit = count_vectorizer.transform(data_fit['title'])
print(X_tf_fit.size)
X_tf_blindtest = count_vectorizer.transform(data_blindtest['title'])
print(X_tf_blindtest.size)
lda = LatentDirichletAllocation(n_components=500, random_state=0)
lda.fit(X_tf_fit)
X_lda_fit = lda.transform(X_tf_fit)
X_lda_fit_blindtest = lda.transform(X_tf_blindtest)
print(X_lda_fit.size)
print(X_lda_fit_blindtest.size)
gbm_model_with_lda = lgb.LGBMClassifier()

# print result
precision_cv_score = model_selection.cross_val_score(gbm_model_with_lda, X_lda_fit, y_fit, cv=5,
                                                     n_jobs=-2, scoring='precision_macro').mean()
recall_cv_score = model_selection.cross_val_score(gbm_model_with_lda, X_lda_fit, y_fit, cv=5,
                                                  n_jobs=-2, scoring='recall_macro').mean()
f1_cv_score = model_selection.cross_val_score(gbm_model_with_lda, X_lda_fit, y_fit, cv=5, n_jobs=-2,
                                              scoring='f1_macro').mean()
