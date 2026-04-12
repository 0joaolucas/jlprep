# preprocess/exploration.py
from pandas.api.types import (
    is_numeric_dtype,
    is_categorical_dtype,
    is_object_dtype,
    is_bool_dtype,
)

__all__ = ["check_class_balance"]

def check_class_balance(data, target, plot=False):

    """
    Analisa o balanceamento das classes em uma variável alvo de um dataset.

    Esta função calcula a distribuição das classes, identifica o tipo de problema
    (classificação binária ou multiclasse), mede o nível de desbalanceamento por meio
    da razão entre a classe majoritária e minoritária (imbalance ratio) e fornece
    um diagnóstico com sugestões de tratamento.

    Parâmetros
    ----------
    data : pandas.DataFrame
        Dataset contendo a variável alvo.

    target : str
        Nome da coluna que representa a variável alvo.

    plot : bool, opcional (default=False)
        Se True, exibe um gráfico de barras com a distribuição das classes.

    Retorna
    -------
    dict
        Um dicionário contendo:
        - 'n_classes' : int
            Número de classes distintas.
        - 'problem_type' : str
            Tipo de problema ('Binary classification' ou 'Multiclass classification').
        - 'imbalance_ratio' : float
            Razão entre a classe majoritária e minoritária.
        - 'status' : str
            Nível de desbalanceamento do dataset.
        - 'suggestion' : str
            Sugestão de tratamento baseada no nível de desbalanceamento.

    Levanta
    -------
    KeyError
        Se a coluna especificada em 'target' não existir no dataset.

    ValueError
        Se a variável alvo possuir menos de duas classes.

    Notas
    -----
    - Valores nulos na variável alvo são automaticamente ignorados.
    - O diagnóstico de desbalanceamento é baseado nos seguintes critérios:
        * ≤ 1.5  → Dataset balanceado
        * ≤ 3    → Levemente desbalanceado
        * ≤ 10   → Moderadamente desbalanceado
        * > 10   → Altamente desbalanceado

    Exemplos
    --------
    >>> check_class_balance(df, target='classe')
    >>> check_class_balance(df, target='target', plot=True)
    """

    import pandas as pd

    #verifica se a coluna existe
    if not target in data.columns:
        raise KeyError("A coluna ", target, " não existe no dataset")

    col = data[target].dropna()

    #contagem e porcentagem de ocorrências de cada classe
    class_count = col.value_counts()
    class_proportion = col.value_counts(normalize=True) * 100

    #numero de classes
    n_class = len(class_count)

    #tipo de problema
    problem = None
    if n_class == 2:
        problem = "Binary classification"
    elif n_class > 2:
        problem = "Multiclass classification"
    else:
        raise ValueError("O dataset precisar ter ao menos 2 classes")
    

    #descobrindo maior e menor
    max_class = class_count.max()
    min_class = class_count.min()

    imbalance_ratio = max_class / min_class

    if imbalance_ratio <= 1.5:
        status = "Dataset balanceado"
        suggestion = "Nenhuma ação necessária"
    elif imbalance_ratio <= 3:
        status = "Dataset levemente desbalanceado"
        suggestion = "Monitorar o desempenho do modelo"
    elif imbalance_ratio <= 10:
        status = "Dataset moderadamente desbalanceado"
        suggestion = "Considerar técnicas de balanceamento"
    else:
        status = "Dataset altamente desbalanceado"
        suggestion = "Aplicar técnicas de balanceamento (SMOTE, oversampling, etc.)"

    # Print final
    print("Distribuição das classes:\n")

    for classe in class_count.index:
        print(f"{classe}: {class_count[classe]} ({class_proportion[classe]:.2f}%)")

    print("\nNúmero de classes:", n_classes)
    print("Tipo de problema:", problem)

    print(f"\nImbalance ratio: {imbalance_ratio:.2f}")
    print("Status:", status)
    print("Sugestão:", suggestion)

    # 9 gráfico
    if plot:
        class_count.plot(kind="bar", title="Distribuição das classes")

    # 10 Retorno
    return {
        "n_classes": n_classes,
        "problem_type": problem,
        "imbalance_ratio": imbalance_ratio,
        "status": status,
        "suggestion": suggestion
    }