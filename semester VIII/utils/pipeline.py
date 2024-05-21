import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from spacy.lang.ru.stop_words import STOP_WORDS

class PrepInteractionPipeline():
    def transform(self, df: pd.DataFrame):
        df.dropna(subset='total_dur', inplace=True)
        df.drop(['total_dur'], axis=1, inplace=True)
        df['watched_pct'] = df['watched_pct'].fillna(50.)

class PrepMoviePipeline():
    def __init__(self, keyword_features_count: int):
        def __tokenize__(line: str):
            return [word for word in line.split(', ')
            if not word.isnumeric() and word not in {'', 'nan'}]
        
        self.kw_vectorizer = TfidfVectorizer(
            tokenizer=__tokenize__, token_pattern=None,
            max_features=keyword_features_count, stop_words=list(STOP_WORDS))
        self.genres_vectorizer = TfidfVectorizer(
            tokenizer=__tokenize__, token_pattern=None)
        
        self.age_rating_ohe = OneHotEncoder(sparse_output=False)

    def __transform_df__(self, df: pd.DataFrame):
        df['age_rating'] = df['age_rating'].fillna(0).astype(int).astype('category')
        df.drop(['title_orig', 'release_year', 'countries', 
            'for_kids', 'studios', 'directors', 'actors', 'description'],
            axis=1, inplace=True)
    
    def __transform_X__(self, df: pd.DataFrame,
        X_genres: np.ndarray, X_age_rating: np.ndarray, X_keywords: np.ndarray):
        
        X_content_type = (df['content_type'] == 'film').\
            astype(int).values.reshape(-1, 1)

        return np.hstack([X_content_type, X_genres, X_age_rating, X_keywords])

    def fit_transform(self, df: pd.DataFrame):
        self.__transform_df__(df)

        X_genres = self.genres_vectorizer.fit_transform(
            df['genres'].values.astype('U')).todense()
        X_age_rating = self.age_rating_ohe.fit_transform(
            df['age_rating'].values.astype('U').reshape(-1, 1))
        X_keywords = self.kw_vectorizer.fit_transform(
            df['keywords'].values.astype('U')).todense()
        
        return self.__transform_X__(df, X_genres, X_age_rating, X_keywords)
    
    def transform(self, df: pd.DataFrame):
        self.__transform_df__(df)

        X_genres = self.genres_vectorizer.transform(
            df['genres'].values.astype('U')).todense()
        X_age_rating = self.age_rating_ohe.transform(
            df['age_rating'].values.astype('U').reshape(-1, 1))
        X_keywords = self.kw_vectorizer.transform(
            df['keywords'].values.astype('U')).todense()
        
        return self.__transform_X__(df, X_genres, X_age_rating, X_keywords)
        
class PrepUserPipeline():
    def __init__(self):
        self.age_ohe = OneHotEncoder(sparse_output=False)
        self.income_ohe = OneHotEncoder(sparse_output=False)
    
    def __transform_df__(self, df: pd.DataFrame):
        df['kids_flg'] = df['kids_flg'].fillna(0)
        df['age'] = df['age'].fillna('age_25_34')
        df['income'] = df['income'].fillna('income_20_40')
        df['sex'] = df['sex'].fillna('Ж')

    def __transform_X__(self, df: pd.DataFrame,
        X_age: np.ndarray, X_income: np.ndarray):

        X_sex = (df['sex'] == 'М').astype(int).values.reshape(-1, 1)
        X_kids_flag = df['kids_flg'].to_numpy().reshape(-1, 1)

        return np.hstack([X_age, X_income, X_sex, X_kids_flag])

    def fit_transform(self, df: pd.DataFrame):
        self.__transform_df__(df)

        X_age = self.age_ohe.fit_transform(
            df['age'].values.astype('U').reshape(-1, 1))
        X_income = self.income_ohe.fit_transform(
            df['income'].values.astype('U').reshape(-1, 1))
        
        return self.__transform_X__(df, X_age, X_income)
    
    def transform(self, df: pd.DataFrame):
        self.__transform_df__(df)

        X_age = self.age_ohe.transform(
            df['age'].values.astype('U').reshape(-1, 1))
        X_income = self.income_ohe.transform(
            df['income'].values.astype('U').reshape(-1, 1))
        
        return self.__transform_X__(df, X_age, X_income)