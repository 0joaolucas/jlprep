# jlprep

Biblioteca em Python para **análise e pré-processamento de dados** voltada para aplicações de Machine Learning.

A `jlprep` foi desenvolvida como parte de um **Trabalho de Conclusão de Curso (TCC)**, com foco na análise e melhoria da qualidade de dados utilizados em modelos de Machine Learning.

---

## 📌 Objetivo

A biblioteca busca centralizar etapas comuns do pré-processamento de dados, como:

- Limpeza de dados  
- Codificação de variáveis categóricas  
- Análise exploratória  
- Avaliação da qualidade dos dados  

Além disso, fornece **diagnósticos interpretáveis**, auxiliando na tomada de decisão antes do treinamento de modelos de Machine Learning.

---

## 🎓 Contexto do TCC

Este projeto foi desenvolvido no contexto do tema:

> **Qualidade dos dados em Machine Learning (dados faltantes, desbalanceados e ruidosos)**

A proposta da `jlprep` é oferecer uma abordagem prática para identificação e tratamento desses problemas, reunindo funcionalidades essenciais em uma única biblioteca.

---

## ⚙️ Funcionalidades

### 🔹 Limpeza de dados
- Remoção de valores duplicados  
- Tratamento de valores faltantes  
- Remoção de outliers  

### 🔹 Transformação
- Label Encoding  
- One-Hot Encoding  

### 🔹 Exploração de dados
- Resumo estatístico do dataset  
- Visualização da distribuição de valores  

### 🔹 Qualidade dos dados
- Verificação de inconsistências  
- Análise de desbalanceamento de classes  
- Sugestões de pré-processamento  

---

## 📊 Exemplo de uso

```python
from jlprep import check_class_balance

resultado = check_class_balance(df, target="classe", plot=True)

print(resultado)