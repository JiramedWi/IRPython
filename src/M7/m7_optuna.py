import joblib
import numpy as np
import optuna
import lightgbm as lgb
from scipy.sparse import hstack
from sklearn import metrics, model_selection
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd

data_texts = joblib.load('resource/data_texts.pkl')
y = joblib.load('resource/y.pkl')
data_fit, data_blindtest, y_fit, y_blindtest = model_selection.train_test_split(data_texts, y, test_size=0.1)
data_fit_train, data_fit_test, y_fit_train, y_fit_test = model_selection.train_test_split(data_fit, y_fit,
                                                                                          test_size=0.3)
print('start tfidf')
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1))
cleaned_title = data_fit['title']
cleaned_body = data_fit['body']
tfidf_vectorizer.fit(cleaned_title + cleaned_body)
X_tfidf_fit = tfidf_vectorizer.transform(data_fit['title'])

X_tfidf_fit_train = tfidf_vectorizer.transform(data_fit_train['title'])
X_tfidf_fit_test = tfidf_vectorizer.transform(data_fit_test['title'])
X_tfidf_blindtest = tfidf_vectorizer.transform(data_blindtest['title'])

print('start lsa')
lsa = TruncatedSVD(n_components=500, n_iter=100, random_state=0)
lsa.fit(X_tfidf_fit)
X_lsa_fit = lsa.transform(X_tfidf_fit)

X_fit_with_lsa = hstack([X_tfidf_fit, X_lsa_fit]).tocsr()


print('start lda')
count_vectorizer = CountVectorizer(ngram_range=(1, 1))
count_vectorizer.fit(cleaned_title + cleaned_body)
X_tf_fit = count_vectorizer.transform(data_fit['title'])
X_tf_blindtest = count_vectorizer.transform(data_blindtest['title'])
lda = LatentDirichletAllocation(n_components=500, random_state=0)
lda.fit(X_tf_fit)
X_lda_fit = lda.transform(X_tf_fit)

X_fit_with_lda = hstack([X_tfidf_fit, X_lda_fit]).tocsr()


def objective(trial):
    dtrain = lgb.Dataset(X_fit_with_lda, label=y_fit)

    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(X_tfidf_fit_test)
    pred_labels = np.rint(preds)
    accuracy = metrics.roc_auc_score(y_fit_test, pred_labels)
    return accuracy


# # Find best parameters
print('start optuna')
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

best_params = study.best_params
print(best_params)

param_tfidf = {'lambda_l1': 1.2858081316484365e-05,
               'lambda_l2': 8.354342479733495e-05,
               'num_leaves': 180,
               'feature_fraction': 0.9345997019195788,
               'bagging_fraction': 0.9185800518921354,
               'bagging_freq': 6,
               'min_child_samples': 19}
gbm_model = lgb.LGBMClassifier(**param_tfidf)

df_result_cv = pd.DataFrame(columns=['precision_cv_score', 'recall_cv_score', 'f1_cv_score'])

precision_cv_score = model_selection.cross_val_score(gbm_model, X_tfidf_fit, y_fit, cv=5,
                                                     n_jobs=-2, scoring='precision_macro').mean()
recall_cv_score = model_selection.cross_val_score(gbm_model, X_tfidf_fit, y_fit, cv=5, n_jobs=-2,
                                                  scoring='recall_macro').mean()
f1_cv_score = model_selection.cross_val_score(gbm_model, X_tfidf_fit, y_fit, cv=5, n_jobs=-2,
                                              scoring='f1_macro').mean()

df_result_cv.append({'precision_cv_score': precision_cv_score, 'recall_cv_score': recall_cv_score,
                     'f1_cv_score': f1_cv_score}, ignore_index=True)
joblib.dump(df_result_cv, 'resource/df_result_tfidf_cv.pkl')
print('CV: p:{0:.4f} r:{1:.4f} f:{2:.4f}'.format(precision_cv_score, recall_cv_score,
                                                 f1_cv_score))

# # Test the model
df_result_test = pd.DataFrame(columns=['precision_test_score', 'recall_test_score', 'f1_test_score'])
gbm_model.fit(X_tfidf_fit_train, y_fit_train, eval_set=[(X_tfidf_fit_test, y_fit_test)],
              eval_metric='AUC')

precision_test_score = metrics.precision_score(gbm_model.predict(X_tfidf_blindtest), y_blindtest,
                                               average='macro')
recall_test_score = metrics.recall_score(gbm_model.predict(X_tfidf_blindtest), y_blindtest,
                                         average='macro')
f1_test_score = metrics.f1_score(gbm_model.predict(X_tfidf_blindtest), y_blindtest,
                                 average='macro')

df_result_test.append({'precision_test_score': precision_test_score, 'recall_test_score': recall_test_score, 'f1_test_score': f1_test_score}, ignore_index=True)
joblib.dump(df_result_test, 'resource/df_result_tfidf_test.pkl')
print('test: p:{0:.4f} r:{1:.4f} f:{2:.4f}'.format(precision_test_score, recall_test_score,
                                                   f1_test_score))
