""" Pré-processamento, limpeza, e remoção de dados.
"""
from sklearn.model_selection import train_test_split
from unidecode import unidecode

import pandas as pd


def dataprep(df):
    """ Remove colunas com nulos acima de 50%.
        Remove linhas com valores nulos nas colunas:
          - city
          - collateral_value
          - monthly_payment
          - informed_purpose.
        Finalmente, preenche com zero os nulos
        da coluna collateral_debt.
    """
    filtro_pre_aprovados = (df.pre_approved == 1.0)
    n_pre_aprovados = df[filtro_pre_aprovados].shape[0]
    thresh = 0.5 * n_pre_aprovados
    subset = [
        'city',
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

    """ Tratamento do city
    """
    publico['city'] = publico.city.apply(
        lambda x: unidecode(x.lower().replace(" ", ""))
    )

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
    publico = publico[publico.auto_year > '2002.0']

    publico = publico[publico.collateral_net_value > 0.0]

    publico_n = publico.select_dtypes(include=['float64'])
    quantiles = publico_n.quantile(.995)
    for key, value in quantiles.iteritems():
        filtro = publico[key] < value
        publico = publico[filtro]

    cities = publico.city.value_counts()
    bad_cities = cities[cities < 9].index
    publico = publico[~publico.city.isin(bad_cities)]

    auto_brands = publico.auto_brand.value_counts()
    bad_auto_brands = auto_brands[auto_brands < 6].index
    publico = publico[~publico.auto_brand.isin(bad_auto_brands)]

    auto_models = publico.auto_model.value_counts()
    bad_auto_models = auto_models[auto_models < 8].index
    publico = publico[~publico.auto_model.isin(bad_auto_models)]

    landing_pages = publico.landing_page.value_counts()
    bad_landing_pages = landing_pages[landing_pages < 5].index
    publico = publico[~publico.landing_page.isin(bad_landing_pages)]

    publico = publico[publico.landing_page_product != 'HomeFin']
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
