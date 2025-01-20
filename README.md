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

## Passo a Passo

1. **Instalação de dependências:**
   ```bash
   pip install pandas scikit-learn
