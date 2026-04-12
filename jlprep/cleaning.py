__all__ = ["remove_duplicates",
           "remove_outliers",
           "clean_missing"]

def remove_duplicates(data, visualization=False, remove=False):
    """
    Remove ou visualiza duplicatas de um dataframe.

    :param data: Dataframe com os dados a serem analisados.
    :param visualization: Se True, visualiza as duplicatas no Dataframe.
    :param remove: Se True, remove as duplicatas no Dataframe.
    :return: Dataframe com ou sem duplicatas, dependendo do parâmetro 'remove'
    
    """
    #Verificando o comportamento
    if visualization and remove:
        print("Exibindo duplicatas")
        print(data[data.duplicated()])
        print("\nRemovendo duplicatas...")
        return data.drop_duplicates()
        
    elif visualization:
        
        print("Exibindo duplicatas:")
        return data[data.duplicated()]

    elif remove:
       
        print("Removendo duplicatas...")
        return data.drop_duplicates()

    else:
        
        print("Nenhuma ação foi realizada. Use 'visualization' para visualizar ou 'remove' para remover duplicatas.")
        return data



def remove_outliers(data, method='iqr', threshold=1.5, columns=None):

    """
    Remove outliers de um DataFrame com base em métodos estatísticos.

    Parâmetros
    ----------
    data : pandas.DataFrame
        Conjunto de dados a ser processado.
    method : str, opcional (default='iqr')
        Método usado para detecção de outliers. Pode ser:
        - 'iqr': usa o intervalo interquartil (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        - 'zscore': usa o desvio padrão em relação à média.
    threshold : float, opcional (default=1.5)
        Fator usado para definir o limite dos outliers.
        - Para 'iqr': multiplicador do IQR.
        - Para 'zscore': valor absoluto do Z-score máximo permitido.
    columns : list ou None, opcional
        Lista de colunas numéricas a serem analisadas. Se None, aplica em todas as colunas numéricas.

    Retorna
    -------
    pandas.DataFrame
        DataFrame sem os outliers detectados.

    Lança
    -----
    ValueError
        Se o método informado não for reconhecido.
    """

    # Seleciona as colunas numéricas
    if columns is None:
        columns = data.select_dtypes(include=["number"]).columns

    
    # Verifica o método escolhido
    match method:
        case "iqr":
            print("Removendo outliers usando o método IQR")

            for col in columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_limit = Q1 - threshold * IQR
                upper_limit = Q3 + threshold * IQR

                data = data[(data[col] >= lower_limit) & data[col] <= upper_limit]

                return data
            

        case "zscore":
            print("Removendo outliers usando o método ZSCORE")

            for col in columns:
                mean = data[col].mean()
                std = data[col].std()

                z_scores = (data[col] - mean) / std

                data = data[z_scores.abs() <= threshold]

                return data
            
        case _:
            raise ValueError(f"Método '{method}' não reconhecido. Use 'iqr' ou 'zscore'.")
        



def clean_missing(data, strategy='drop', columns=None, value=None, threshold=None, visualize=False):
    
    """
    Trata valores ausentes em um DataFrame com diferentes estratégias.

    Parâmetros:
    - data: DataFrame do pandas.
    - strategy: Estratégia de tratamento. Pode ser 'drop', 'mean', 'median', 'mode', 'value' ou 'interpolate'.
    - columns: Lista de colunas a serem tratadas. Se None, serão escolhidas com base na estratégia.
    - value: Valor específico para preenchimento (usado apenas se strategy='value').
    - threshold: Proporção de valores ausentes para remover colunas (usado apenas com strategy='drop').
    - visualize: Se True, exibe o resumo de valores ausentes antes do tratamento.

    Retorna:
    - DataFrame com valores ausentes tratados.
    """


    if columns is None:
        if strategy in ["mean", "median", "zscore"]:
            columns = data.select_dtypes(include=["number"]).columns
        elif strategy in ["mode", "value", "drop", "interpolate"]:
            columns = data.columns
        else:
            raise ValueError(f"Estratégia '{strategy}' não reconhecida.")

    match strategy:
        case "drop":
            if threshold is not None:
                
                data = data.loc[:, data.isnull().mean() < threshold]
            else:
                
                data = data.dropna(subset=columns)

        case "mean":
            for col in columns:
                data[col] = data[col].fillna(data[col].mean())

        case "median":
            for col in columns:
                data[col] = data[col].fillna(data[col].median())

        case "mode":
            for col in columns:
                mode = data[col].mode()
                if not mode.empty:
                    data[col] = data[col].fillna(mode[0])

        case "value":
            if value is None:
                raise ValueError("Para a estratégia 'value', o parâmetro 'value' deve ser fornecido.")
            for col in columns:
                data[col] = data[col].fillna(value)

        case "interpolate":
            data[columns] = data[columns].interpolate()

        case _:
            raise ValueError(f"Estratégia '{strategy}' não reconhecida.")

    return data

            