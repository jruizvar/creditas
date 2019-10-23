""" Pré-processamento, limpeza, e remoção de dados.
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

import functools
import nltk


stopwords = nltk.corpus.stopwords.words('portuguese')


def dataprep(df):
    """ Remove colunas com nulos acima de 50%.
        Remove linhas com valores nulos nas colunas
        collateral_value, monthly_payment e informed_purpose.
        Finalmente, preenche com zero os nulos
        da coluna collateral_debt.
    """
    filtro_pre_aprovados = (df.pre_approved == 1.0)
    n_pre_aprovados = df[filtro_pre_aprovados].shape[0]
    thresh = 0.5 * n_pre_aprovados
    subset = ['collateral_value', 'monthly_payment', 'informed_purpose']
    publico = df[filtro_pre_aprovados] \
        .dropna(axis=1, thresh=thresh) \
        .dropna(axis=0, subset=subset) \
        .fillna(value={'collateral_debt': 0.0})

    """ Criação da variável collateral_net_value.
    """
    publico['collateral_net_value'] = \
        publico.collateral_value - publico.collateral_debt
    publico = publico.drop(['collateral_value', 'collateral_debt'], axis=1)

    """ Determina as colunas categóricas
        segundo o número de valores únicos.
        Codifica as colunas categóricas.
    """
    max_nunique = 200
    categorical_columns = [
        c for c in publico.columns if
        publico[c].nunique() < max_nunique
    ]
    le = LabelEncoder()
    for c in categorical_columns:
        publico[c] = le.fit_transform(publico[c].astype('str'))

    """ Remove outliers nas colunas numéricas.
    """
    publico_n = publico.select_dtypes(include=['float64'])
    quantiles = publico_n.quantile([.01, .99]).T
    for row in quantiles.itertuples():
        filtro1 = publico[row[0]] > row[1]
        filtro2 = publico[row[0]] < row[2]
        publico = publico[filtro1 & filtro2]

    """ Remove coluna city, zip_code e auto_model
    """
    publico = publico.drop(['city', 'zip_code', 'auto_model'], axis=1)
    return publico


def tokenizer(text, token_min_length=1):
    """ tratamento da variável informed_purpose
    """
    tokens = nltk.tokenize.word_tokenize(text, language='portuguese')
    return [
        t.lower() for t in tokens
        if len(t) > token_min_length and
        t.lower() not in stopwords
    ]


def bag_of_words(textos):
    """ Sacola de palavras para o modelo de classificação de texto.
    """
    tokens = textos.apply(tokenizer)
    vocabulary = functools.reduce((lambda a, b: a | b), tokens.apply(set))
    vectorizer = CountVectorizer(
        vocabulary=vocabulary
    )
    vectorizer.fit(textos)
    return vectorizer
