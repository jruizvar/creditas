from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectPercentile
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
    selector = SelectPercentile(
        score_func=chi2,
        percentile=10
    )
    bnb = Pipeline([
        ('vec', vectorizer),
        ('sel', selector),
        ('bnb', BernoulliNB(binarize=None))
    ])
    bnb.fit(X, y)
    return bnb


def clf_dt1(X, y):
    """ Árvore de decisão contínua
    """
    dt1 = Pipeline([
        ('sca', StandardScaler()),
        ('dt1', DecisionTreeClassifier(max_depth=4))
    ])
    dt1.fit(X, y)
    return dt1


def clf_dt2(X, y):
    """ Árvore de decisão categórica
    """
    pass_cols = [
        'id',
        'banking_debts',
        'commercial_debts',
    ]
    toencoder = [
        'zip_code',
        'informed_restriction',
        'form_completed',
        'auto_model',
        'auto_brand',
        'auto_year',
        'channel',
        'landing_page',
        'landing_page_product',
        'gender',
    ]
    ctr = ColumnTransformer(
        [
            ('pass_cols', 'passthrough', pass_cols),
            ('toencoder', OrdinalEncoder(), toencoder)

        ]
    )
    dt2 = Pipeline([
        ('ctr', ctr),
        ('dt2', DecisionTreeClassifier(max_depth=5))
    ])
    dt2.fit(X, y)
    return dt2
