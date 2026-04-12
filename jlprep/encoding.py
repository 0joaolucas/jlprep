__all__ = ["encode_label",
           "encode_onehot"]


def encode_label(data, columns=None, return_mapping=False):

    
    """

    Aplica codificação label encoding em colunas categóricas de um Dataframe.

    Cada valor único de texto é substituído por um número inteiro, facilitando o uso
    de dados categóricos em algoritmos de Machine Learning.

    Parâmetros:
    ----------
    data: pandas.Dataframe
        O DataFrame de entrada que contém os dados a serem codificados

    columns: list (opcional)
        Lista de colunas a serem codificadas. Se "None", todas as colunas do tipo 'object' serão usadas.

    return_mapping : bool, opcional
        Se True, retorna também um dicionário com os mapeamentos de rótulo para número inteiro
        para cada coluna codificada.

    Retorna:
    -------
    pandas.DataFrame
        O DataFrame com as colunas especificadas codificadas.

    dict (opcional)
        Um dicionário contendo os mapeamentos usados para cada coluna, se return_mapping=True.
    """

    if columns is None:
        columns = data.select_dtypes(include="object").columns

    if return_mapping:
        mapping_dict = {}

    for col in columns:
        unique_values = data[col].unique()
        mapping = {label: idx for idx, label in enumerate(unique_values)}
        data[col] = data[col].map(mapping)

        if return_mapping:
            mapping_dict[col] = mapping

    
    if return_mapping:
        return data, mapping_dict
    return data



def encode_onehot(data, columns=None, drop_first=False):

    """
    Codifica variáveis categóricas usando One-Hot Encoding.

    Parâmetros:
    data (DataFrame): O DataFrame contendo os dados.
    columns (list, opcional): Lista de colunas a serem codificadas. Se None, codifica todas as colunas categóricas.
    drop_first (bool, opcional): Se True, remove a primeira coluna codificada para evitar multicolinearidade. Padrão é False.

    Retorna:
    DataFrame: DataFrame com as colunas categóricas substituídas por variáveis binárias (0 ou 1).
    """

    import pandas as pd

    if columns is None:
        columns = data.select_dtypes(include='object').columns

    data_encoded = pd.get_dummies(data, columns=columns, drop_first=drop_first)

    data_encoded = data_encoded.astype(int)

    return data_encoded