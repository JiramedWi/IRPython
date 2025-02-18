import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import seaborn as sns

anime = pd.read_csv('resource/anime.csv')

rating = pd.read_csv('resource/rating_complete.csv')
rating = rating[:1000]
anime_features = ['MAL_ID', 'English name', 'Japanese name', 'Score', 'Genres', 'Popularity',
                  'Members', 'Favorites', 'Watching', 'Completed', 'On-Hold', 'Dropped',
                  'Score-1', 'Score-2', 'Score-3', 'Score-4', 'Score-5',
                  'Score-6', 'Score-7', 'Score-8', 'Score-9', 'Score-10',
                  ]
anime = anime[anime_features]

merged_df = anime.merge(rating, left_on='MAL_ID', right_on='anime_id', how='inner')

genre_names = [
    'Action', 'Adventure', 'Comedy', 'Drama', 'Sci-Fi',
    'Game', 'Space', 'Music', 'Mystery', 'School', 'Fantasy',
    'Horror', 'Kids', 'Sports', 'Magic', 'Romance',
]


def genre_to_category(df):
    d = {name: [] for name in genre_names}

    def f(row):
        genres = row.Genres.split(',')
        for genre in genre_names:
            if genre in genres:
                d[genre].append(1)
            else:
                d[genre].append(0)

    df.apply(f, axis=1)
    genre_df = pd.DataFrame(d, columns=genre_names)
    df = pd.concat([df, genre_df], axis=1)
    return df


def make_anime_feature(df):
    df['Score'] = df['Score'].apply(lambda x: np.nan if x == 'Unknown' else float(x))
    for i in range(1, 11):
        df[f'Score-{i}'] = df[f'Score-{i}'].apply(lambda x: np.nan if x == 'Unknown' else float(x))
    df = genre_to_category(df)
    return df


def make_user_feature(df):
    # add user feature
    df['rating_count'] = df.groupby('user_id')['anime_id'].transform('count')
    df['rating_mean'] = df.groupby('user_id')['rating'].transform('mean')
    return df


def preprocess(merged_df):
    merged_df = make_anime_feature(merged_df)
    merged_df = make_user_feature(merged_df)
    return merged_df


merged_df = preprocess(merged_df)
merged_df = merged_df.drop(['MAL_ID', 'Genres'], axis=1)

fit, blindtest = train_test_split(merged_df, test_size=0.2, random_state=0)
fit_train, fit_test = train_test_split(fit, test_size=0.3, random_state=0)
features = ['Score', 'Popularity', 'Members',
            'Favorites', 'Watching', 'Completed', 'On-Hold', 'Dropped',
            'Score-1', 'Score-2', 'Score-3', 'Score-4', 'Score-5',
            'Score-6', 'Score-7', 'Score-8', 'Score-9', 'Score-10',
            'rating_count', 'rating_mean'
            ]
features += genre_names
user_col = 'user_id'
item_col = 'anime_id'
target_col = 'rating'
fit_train = fit_train.sort_values('user_id').reset_index(drop=True)
fit_test = fit_test.sort_values('user_id').reset_index(drop=True)
blindtest = blindtest.sort_values('user_id').reset_index(drop=True)
# model query data
fit_train_query = fit_train[user_col].value_counts().sort_index()
fit_test_query = fit_test[user_col].value_counts().sort_index()
blindtest_query = blindtest[user_col].value_counts().sort_index()

model = lgb.LGBMRanker(n_estimators=1000, random_state=0, early_stopping_rounds=100,
                       verbose=10)
model.fit(
    fit_train[features],
    fit_train[target_col],
    group=fit_train_query,
    eval_set=[(fit_test[features], fit_test[target_col])],
    eval_group=[list(fit_test_query)],
    eval_at=[1, 3, 5, 10],  # calc validation ndcg@1,3,5,10
)
model.predict(blindtest.iloc[:10][features])
plt.figure(figsize=(10, 7))
df_plt = pd.DataFrame({'feature_name': features, 'feature_importance':
    model.feature_importances_})
df_plt.sort_values('feature_importance', ascending=False, inplace=True)
sns.barplot(x="feature_importance", y="feature_name", data=df_plt)
plt.title('feature importance')


def predict(user_df, top_k, anime, rating):
    user_anime_df = anime.merge(user_df, left_on='MAL_ID', right_on='anime_id')
    user_anime_df = make_anime_feature(user_anime_df)
    excludes_genres = list(np.array(genre_names)[np.nonzero([user_anime_df[genre_names].sum(axis=0)
                                                             <= 1])[1]])
    pred_df = make_anime_feature(anime.copy())
    pred_df = pred_df.loc[pred_df[excludes_genres].sum(axis=1) == 0]
    for col in user_df.columns:
        if col in features:
            pred_df[col] = user_df[col].values[0]

    preds = model.predict(pred_df[features])
    topk_idx = np.argsort(preds)[::-1][:top_k]
    recommend_df = pred_df.iloc[topk_idx].reset_index(drop=True)
    # check recommend
    print('---------- Recommend ----------')
    for i, row in recommend_df.iterrows():
        print(f'{i + 1}: {row["Japanese name"]}:{row["English name"]}')
    print('---------- Rated ----------')
    user_df = user_df.merge(anime, left_on='anime_id', right_on='MAL_ID', how='inner')
    for i, row in user_df.sort_values('rating', ascending=False).iterrows():
        print(f'rating:{row["rating"]}: {row["Japanese name"]}:{row["English name"]}')

    return recommend_df


user_id = 1
user_df = rating.copy().loc[rating['user_id'] == user_id]
user_df = make_user_feature(user_df)
predict(user_df, 10, anime, rating)
