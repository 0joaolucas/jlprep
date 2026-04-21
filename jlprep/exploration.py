# preprocess/exploration.py
from pandas.api.types import (
    is_numeric_dtype,
    is_categorical_dtype,
    is_object_dtype,
    is_bool_dtype,
)

__all__ = ["summarize_data",
           "value_counts_plot",
           "check_data_quality"]

def summarize_data(data):
    """
    Gera um resumo exploratório do DataFrame:
      - Tamanho (linhas/colunas)
      - Tipos de dados por coluna
      - Valores ausentes (contagem e %)
      - Estatísticas por coluna:
          * Numéricas: média, mediana, moda, min, max
          * Categóricas (object/category/bool): moda, nº de categorias únicas, freq. da moda
    Retorna:
      dict: {nome_da_coluna: {estatísticas...}}
    """
    # 1) Tamanho
    print("Número de linhas e colunas:", len(data), "linhas,", len(data.columns), "colunas")

    # 2) Tipos
    print("\nTipos de dados por coluna:")
    print(data.dtypes)

    # 3) Ausentes (contagem e %)
    print("\nValores ausentes por coluna (ordem decrescente):")
    print(data.isnull().sum().sort_values(ascending=False))

    print("\nPorcentagem de valores ausentes por coluna:")
    print(((data.isnull().sum() / len(data)) * 100).round(2))

    # 4) Amostra
    print("\n5 primeiras linhas do DataFrame:")
    print(data.head(5))

    # 5) Estatísticas por coluna
    resumo = {}
    for col in data.columns:
        if is_numeric_dtype(data[col]):
            s = data[col].dropna()
            moda_series = s.mode()
            moda = moda_series.iloc[0] if not moda_series.empty else None
            resumo[col] = {
                "média": s.mean() if not s.empty else None,
                "mediana": s.median() if not s.empty else None,
                "moda": moda,
                "mínimo": s.min() if not s.empty else None,
                "máximo": s.max() if not s.empty else None,
            }
        elif is_categorical_dtype(data[col]) or is_object_dtype(data[col]) or is_bool_dtype(data[col]):
            s = data[col].dropna()
            moda_series = s.mode()
            moda = moda_series.iloc[0] if not moda_series.empty else None
            freq_moda = s.value_counts().iloc[0] if not s.value_counts().empty else 0
            resumo[col] = {
                "moda": moda,
                "n_categorias": s.nunique(),
                "freq_moda": int(freq_moda),
            }
        else:
            # Para tipos como datetime, etc., um resumo simples:
            s = data[col].dropna()
            resumo[col] = {
                "tipo": str(data[col].dtype),
                "n_únicos": s.nunique(),
            }

    # Imprime o resumo
    for col, stats in resumo.items():
        print(f"\nColuna: {col}")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    return resumo


def value_counts_plot(data, column, normalize=False, top_n = None):
     
     import matplotlib.pyplot as plt

     """
    Gera um gráfico de barras com a contagem (ou proporção) dos valores únicos
    em uma coluna categórica.

    Parâmetros
    ----------
    data : pandas.DataFrame
        O DataFrame com os dados.
    column : str
        Nome da coluna categórica a ser analisada.
    normalize : bool, opcional (default=False)
        Se True, mostra proporções (%) em vez de contagens absolutas.
    top_n : int, opcional
        Se definido, mostra apenas as top N categorias mais frequentes.

    Retorno
    -------
    None
        Exibe o gráfico.
    """
     
     #Obtendo frequências
     counts = data[column].value_counts(normalize=normalize)

     if top_n is not None:
         counts = counts.head(top_n)

     counts.plot(kind="bar")
     plt.title(f"Distribuição de valores - {column}")
     plt.xlabel("Categorias")
     plt.ylabel("Frequências (%)" if normalize else "Frequências")



def check_data_quality(data):
    """
    Analisa a qualidade dos dados de um DataFrame.

    Funcionalidades:
    - Valores ausentes (%)
    - Duplicatas por coluna
    - Outliers (IQR) para colunas numéricas
    - Cardinalidade de colunas categóricas (com gráfico)
    
    Parâmetros
    ----------
    data : pandas.DataFrame
        O DataFrame a ser analisado.
    
    Retornos
    -------
    dict
        Dicionários com informações sobre valores ausentes, duplicatas, outliers e cardinalidade.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from pandas.api.types import is_numeric_dtype, is_categorical_dtype

    # Inicializa dicionários
    dict_missing_col = {}
    duplicated_col = {}
    dict_outliers_col = {}
    dict_cardinality = {}

    # Valores ausentes
    for col in data.columns:
        percentual_missing = (data[col].isnull().sum() / len(data[col])) * 100
        dict_missing_col[col] = percentual_missing

    # Duplicatas
    for col in data.columns:
        duplicated_count_col = len(data[col]) - len(data[col].drop_duplicates())
        duplicated_col[col] = duplicated_count_col

    # Outliers (colunas numéricas)
    for col in data.columns:
        if is_numeric_dtype(data[col]):
            Q1 = np.percentile(data[col], 25)
            Q3 = np.percentile(data[col], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_iqr = [x for x in data[col] if x < lower_bound or x > upper_bound]
            dict_outliers_col[col] = outliers_iqr

    # Cardinalidade (colunas categóricas)
    for col in data.columns:
        if is_categorical_dtype(data[col]):
            total_row = len(data[col])
            unique_values = data[col].nunique()
            proportion = (unique_values / total_row) * 100
            dict_cardinality[col] = {
                "total": total_row,
                "unique": unique_values,
                "percentual": proportion
            }

    # Preparar listas para o gráfico
    list_name_col = []
    list_percentual = []
    for col, info in dict_cardinality.items():
        list_name_col.append(col)
        list_percentual.append(info['percentual'])

    # Plot gráfico de cardinalidade
    if list_name_col:
        plt.figure(figsize=(10, 6))
        plt.bar(list_name_col, list_percentual, color="blue")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Percentual de cardinalidade")
        plt.xlabel("Colunas categóricas")
        plt.title("Verificação de cardinalidade")
        plt.show()

        # Mensagem de alerta para alta cardinalidade
        for col, perc in zip(list_name_col, list_percentual):
            if perc > 50:
                print(f"Atenção: coluna '{col}' apresenta alta cardinalidade ({perc:.1f}%), considere descartá-la ou tratá-la.")

    # Retornar todos os dicionários
    return {
        "missing": dict_missing_col,
        "duplicated": duplicated_col,
        "outliers": dict_outliers_col,
        "cardinality": dict_cardinality
    }

