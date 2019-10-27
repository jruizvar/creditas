""" Modelos de classificação
"""
from myutils import RavelTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import pandas as pd


def clf_bnb(X, y):
    """ Bernoulli naive bayes para a
        variável de texto informed_purpose
    """
    vectorizer = CountVectorizer(
        binary=True,
        strip_accents='ascii'
    )
    imputer_vectorizer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='nulo')),
        ('ravel', RavelTransformer()),
        ('vector', vectorizer)
    ])
    ctr = ColumnTransformer(
        [('texto', imputer_vectorizer, ['informed_purpose'])]
    )
    selector = SelectPercentile(
        score_func=chi2,
        percentile=20
    )
    bnb = Pipeline([
        ('ctr', ctr),
        ('sel', selector),
        ('bnb', BernoulliNB(binarize=None))
    ])
    bnb.fit(X, y)
    return bnb


def clf_dt1(X, y):
    """ Árvore de decisão contínua
    """
    toscaler = [
        'monthly_income',
        'loan_amount',
        'monthly_payment',
        'collateral_net_value',
    ]
    imputer_scaler = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    ctr = ColumnTransformer(
        [('toscaler', imputer_scaler, toscaler)]
    )
    dt1 = Pipeline([
        ('ctr', ctr),
        ('dt1', DecisionTreeClassifier(max_depth=4))
    ])
    dt1.fit(X, y)
    return dt1


def clf_dt2(X, y):
    """ Árvore de decisão categórica
    """
    pass_cols = [
        'id',
        'age',
        'zip_code',
        'banking_debts',
        'commercial_debts',
        'auto_year',
    ]
    toencoder = [
        'auto_brand',
        'informed_restriction',
        'form_completed',
        'channel',
        'landing_page',
    ]
    imputer_encoder = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    ctr = ColumnTransformer(
        [
            ('pass_cols', 'passthrough', pass_cols),
            ('toencoder', imputer_encoder, toencoder)
        ]
    )
    dt2 = Pipeline([
        ('ctr', ctr),
        ('dt2', DecisionTreeClassifier(max_depth=7))
    ])
    dt2.fit(X, y)
    return dt2
