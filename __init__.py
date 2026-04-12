# jlprep/__init__.py

__version__ = "0.1.0"

# Importando funções diretamente para facilitar o uso
from .preprocess.cleaning import remove_duplicates, remove_outliers, clean_missing
from .preprocess.encoding import encode_label, encode_onehot
from .preprocess.exploration import summarize_data, value_counts_plot, check_data_quality
from .preprocess.quality import check_class_balance

__all__ = ["remove_duplicates",
           "remove_outliers",
           "clean_missing",
           "encode_label",
           "encode_onehot",
           "summarize_data",
           "value_counts_plot",
           "check_class_balance",
           "check_data_quality"]


"""
JlPrep - Biblioteca de pré-processamento de dados para Machine Learning

Desenvolvida por João Lucas para facilitar tarefas como:
- Limpeza de dados
- Tratamento de valores ausentes
- Codificação de variáveis categóricas
- Análise exploratória (em futuras versões)
"""
