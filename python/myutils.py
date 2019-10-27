""" Pré-processamento, limpeza, e remoção de dados.
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

import pandas as pd


class RavelTransformer(BaseEstimator, TransformerMixin):
    """ Converte uma variável 2D em 1D
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.ravel()


def dataprep(df):
    """ Método para efetuar o pré-processamento dos dados
    """
    print('Antes do dataprep:', df.shape)

    """ Definição do público de modelagem
    """
    pre_aprovados = (df.pre_approved == 1.0)
    publico = df[pre_aprovados]

    """ Tratamento do campo zip_code
    """
    bad_zip_codes = publico.zip_code.apply(lambda x: x.find('X')) < 4
    publico = publico[~bad_zip_codes]
    publico['zip_code'] = publico.zip_code.apply(lambda x: int(x[:4]))

    """ Casting de colunas
    """
    publico['auto_year'] = publico.auto_year.astype(int)
    publico['sent_to_analysis'] = publico.sent_to_analysis.astype(int)

    """ Tratamento de outliers
    """
    auto_brands = publico.auto_brand.value_counts()
    rare_auto_brands = auto_brands[auto_brands < 5].index
    publico = publico[~publico.auto_brand.isin(rare_auto_brands)]

    landing_pages = publico.landing_page.value_counts()
    rare_landing_pages = landing_pages[landing_pages < 5].index
    publico = publico[~publico.landing_page.isin(rare_landing_pages)]

    print('Final do dataprep:', publico.shape)
    return publico


def amostragem(df, target):
    """ Separação da amostra:
          - df_valid (20%)
          - df_teste (20%)
          - df_train (60%)
    """
    df_dummy, df_valid = train_test_split(
        df, test_size=0.20, random_state=42)

    df_train, df_teste = train_test_split(
        df_dummy, test_size=0.25, random_state=42)

    return df_train, df_teste, df_valid
