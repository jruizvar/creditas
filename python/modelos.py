from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import pandas as pd


def clf_knn(X_train, y_train):
    """ K-nearest neighbors
    """
    param_grid = {
        'n_neighbors': range(1, 50, 10)
    }
    knn = GridSearchCV(
        KNeighborsClassifier(p=1),
        param_grid,
        cv=2
    )
    X_train_knn = X_train.zip_code.values.reshape(-1, 1)
    knn.fit(X_train_knn, y_train)
    return knn


def clf_bnb(X_train, y_train):
    """ Bernoulli naive bayes
    """
    vectorizer = CountVectorizer(
        binary=True,
        strip_accents='ascii'
    )
    bnb = Pipeline([
        ('vec', vectorizer),
        ('bnb', BernoulliNB(binarize=None))
    ])
    X_train_bnb = X_train.informed_purpose
    bnb.fit(X_train_bnb, y_train)
    return bnb


def clf_gnb(X_train, y_train):
    """ Gaussian naive bayes
    """
    gnb = Pipeline([
        ('sca', StandardScaler()),
        ('gnb', GaussianNB())
    ])
    X_train_gnb = X_train.select_dtypes(include=['float64'])
    gnb.fit(X_train_gnb, y_train)
    return gnb


def clf_dtc(X_train, y_train):
    """ Árvore de decisão
    """
    dtc = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )
    X_train_dtc = X_train.select_dtypes(include=['int64'])
    dtc.fit(X_train_dtc, y_train)
    return dtc
