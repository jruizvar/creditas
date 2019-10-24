""" Pré-processamento, limpeza, e remoção de dados.
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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

    """ Remove colunas
    """
    drop_columns = [
        'verified_restriction',
        'expired_debts',
        'pre_approved',
        # 'zip_code',
    ]
    publico = publico.drop(drop_columns, axis=1)

    """ Tratamento do zip_code
    """
    publico['zip_code'] = publico.zip_code.apply(lambda x: x.replace('X', ''))

    """ Criação da variável collateral_net_value.
    """
    publico['collateral_net_value'] = \
        publico.collateral_value - publico.collateral_debt
    publico = publico.drop(['collateral_value', 'collateral_debt'], axis=1)

    """ Codifica as colunas categóricas.
    """
    categorical_columns = [
        # 'city',
        # 'state',
        # 'dishonored_checks',
        # 'banking_debts',
        # 'commercial_debts',
        # 'protests',
        # 'marital_status',
        'informed_restriction',
        # 'loan_term',
        # 'auto_brand',
        'auto_model',
        'auto_year',
        'form_completed',
        'sent_to_analysis',
        'channel',
        'landing_page',
        'landing_page_product',
        'gender',
        # 'utm_term',
        'education_level',
    ]
    le = LabelEncoder()
    for c in categorical_columns:
        publico[c] = le.fit_transform(publico[c].astype(str))

    """ Remove outliers nas colunas numéricas.
    """
    publico = publico[publico.collateral_net_value > 0.0]
    publico_n = publico.select_dtypes(include=['float64'])
    quantiles = publico_n.quantile(.995)
    for key, value in quantiles.iteritems():
        filtro = publico[key] < value
        publico = publico[filtro]

    return publico


def amostragem(df, target):
    """ Amostras de treino, teste e validação
    """
    y = df[target]
    X = df.drop(target, axis=1)

    X_train, X_teste, y_train, y_teste = train_test_split(
                     X, y, test_size=0.5, random_state=42)
    X_teste, X_valid, y_teste, y_valid = train_test_split(
         X_teste, y_teste, test_size=0.5, random_state=42)

    return X_train, X_teste, X_valid, y_train, y_teste, y_valid
