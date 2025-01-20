# Machine Learning para Análise de Score de Crédito

Este projeto utiliza algoritmos de Machine Learning para prever o score de crédito de clientes com base em um conjunto de dados fornecido. Ele foi implementado em Python e conta com ferramentas da biblioteca `scikit-learn` para treinamento e avaliação dos modelos.

## Tecnologias Utilizadas

- **Python**
- **Pandas**: Manipulação e análise de dados
- **Scikit-learn**: Treinamento, avaliação e implementação de algoritmos de Machine Learning

## Objetivo

Prever o score de crédito dos clientes em categorias:
- **Good (Boa)**
- **Standard (OK)**
- **Poor (Ruim)**

Os dados são preparados para a aplicação de algoritmos como Random Forest e K-Nearest Neighbors (KNN). O melhor modelo é utilizado para prever o score de crédito de novos clientes.

## Estrutura do Código

### 1. Entendimento dos Dados
O conjunto de dados inicial, `clientes.csv`, é carregado e inspecionado para identificar informações relevantes.

### 2. Preparação dos Dados
Os dados categóricos são codificados para valores numéricos utilizando o `LabelEncoder`:
- **Profissão**
- **Mix de crédito**
- **Comportamento de pagamento**

### 3. Divisão dos Dados
Os dados são divididos em conjuntos de treino e teste:
- **X**: Variáveis preditoras (sem `score_credito` e `id_cliente`)
- **Y**: Variável alvo (`score_credito`)

### 4. Treinamento dos Modelos
Dois modelos de Machine Learning foram testados:
- **Random Forest Classifier**
- **K-Nearest Neighbors (KNN)**

### 5. Avaliação dos Modelos
Os modelos são avaliados pela acurácia utilizando o conjunto de teste. O modelo com maior acurácia é selecionado para previsão de novos clientes.

### 6. Previsão para Novos Clientes
O conjunto de dados `novos_clientes.csv` é utilizado para previsões com o melhor modelo treinado.

## Como Usar

1. **Instale as dependências:**
   ```bash
   pip install pandas scikit-learn

2. **Execute o código abaixo:**
# Passo 0 - Entender a empresa e o desafio da empresa
# Passo 1 - Importar a base de dados
import pandas as pd

tabela = pd.read_csv("clientes.csv")
display(tabela)

# Score de crédito = Nota de crédito
# Good = Boa
# Standard = OK
# Poor = Ruim

# Passo 2 - Preparar a base de dados para a Inteligência Artificial
display(tabela.info())
from sklearn.preprocessing import LabelEncoder

# Codificação de colunas categóricas
codificador_profissao = LabelEncoder()
tabela['profissao'] = codificador_profissao.fit_transform(tabela['profissao'])

codificador_credito = LabelEncoder()
tabela['mix_credito'] = codificador_credito.fit_transform(tabela['mix_credito'])

codificador_pagamento = LabelEncoder()
tabela['comportamento_pagamento'] = codificador_pagamento.fit_transform(tabela['comportamento_pagamento'])
display(tabela.info())

# Separar X e Y
y = tabela['score_credito']
x = tabela.drop(columns=["score_credito", "id_cliente"])

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

# Treinar modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

modelo_arvoredecisao = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()

modelo_arvoredecisao.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)

# Avaliação
from sklearn.metrics import accuracy_score
print("Acurácia Random Forest:", accuracy_score(y_teste, modelo_arvoredecisao.predict(x_teste)))
print("Acurácia KNN:", accuracy_score(y_teste, modelo_knn.predict(x_teste)))

# Previsão para novos clientes
tabela_novos_clientes = pd.read_csv("novos_clientes.csv")
tabela_novos_clientes['profissao'] = codificador_profissao.transform(tabela_novos_clientes['profissao'])
tabela_novos_clientes['mix_credito'] = codificador_credito.transform(tabela_novos_clientes['mix_credito'])
tabela_novos_clientes['comportamento_pagamento'] = codificador_pagamento.transform(tabela_novos_clientes['comportamento_pagamento'])

nova_previsao = modelo_arvoredecisao.predict(tabela_novos_clientes)
print("Previsão para novos clientes:", nova_previsao)
