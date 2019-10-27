""" Pré-processamento, limpeza, e remoção de dados.
"""
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd


def dataprep(df):
    """ Remove colunas com nulos acima de 50%.
        Remove linhas com valores nulos nas colunas:
          - collateral_value
          - monthly_payment
          - informed_purpose.
        Finalmente, preenche com zero os nulos
        da coluna collateral_debt.
    """
    print('Shape antes do dataprep:', df.shape)

    filtro_pre_aprovados = (df.pre_approved == 1.0)
    n_pre_aprovados = df[filtro_pre_aprovados].shape[0]
    thresh = 0.5 * n_pre_aprovados
    subset = [
        'collateral_value',
        'monthly_payment',
        'informed_purpose',
    ]
    publico = df[filtro_pre_aprovados] \
        .dropna(axis=1, thresh=thresh) \
        .dropna(axis=0, subset=subset) \
        .fillna(value={'collateral_debt': 0.0})

    """ Remove colunas
    """
    drop_columns = [
        'verified_restriction',
        'expired_debts',
        'pre_approved'
    ]
    publico = publico.drop(drop_columns, axis=1)

    """ Tratamento do campo zip_code
    """
    bad_zip_codes = publico.zip_code.apply(lambda x: x.find('X')) < 4
    publico = publico[~bad_zip_codes]
    publico['zip_code'] = publico.zip_code.apply(lambda x: x[:2])

    """ Criação da variável collateral_net_value.
    """
    publico['collateral_net_value'] = \
        publico.collateral_value - publico.collateral_debt
    publico = publico.drop(['collateral_value', 'collateral_debt'], axis=1)

    """ Casting de colunas
    """
    categorical_columns = [
        'informed_restriction',
        'auto_model',
        'auto_year',
        'form_completed',
        'sent_to_analysis',
        'channel',
        'landing_page',
        'landing_page_product',
        'gender',
        'education_level',
    ]
    for c in categorical_columns:
        publico[c] = publico[c].astype(str)

    """ Remove outliers
    """
    publico = publico[np.abs(publico.collateral_net_value) < 3.e5]

    publico = publico[publico.monthly_income < 3.e5]

    publico = publico[~publico.zip_code.isin(['10'])]

    publico = publico[~publico.auto_brand.isin(['90', '92', '99'])]

    publico = publico[publico.auto_year > '2002.0']

    auto_models = publico.auto_model.value_counts()
    bad_auto_models = auto_models[auto_models < 7].index
    publico = publico[~publico.auto_model.isin(bad_auto_models)]

    landing_pages = publico.landing_page.value_counts()
    bad_landing_pages = landing_pages[landing_pages < 6].index
    publico = publico[~publico.landing_page.isin(bad_landing_pages)]

    print('Shape no final do dataprep:', publico.shape)
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
