# Model Card – Predição de Evasão Estudantil

Este projeto visa prever a evasão estudantil com base em dados do ENADE (ENAPE), utilizando técnicas de aprendizado de máquina. A modelagem foi realizada com um classificador Random Forest, após extensiva análise exploratória dos dados.

## Detalhes do Projeto

O modelo foi implementado por [Seu Nome] e utiliza a linguagem Python, com as bibliotecas `pandas`, `matplotlib`, `seaborn`, `scikit-learn` e `numpy`. A pipeline de desenvolvimento inclui:

- Análise exploratória com gráficos (histogramas, boxplots, mapa de calor, regressões)
- Engenharia de atributos
- Tratamento de valores ausentes
- Codificação de variáveis categóricas
- Normalização dos dados
- Treinamento com `RandomForestClassifier`
- Avaliação por métricas de classificação

## Scripts Utilizados

- `analise_exploratoria_graficos.py`: Geração de estatísticas descritivas e visualizações para entendimento dos dados.
- `analise_evasao.py`: Pipeline de pré-processamento, criação de variável-alvo e treinamento/avaliação do modelo.
- `main.py`: Versão alternativa da análise exploratória com regressão linear e logística.
- Dataset: `enape_raw_dataset.csv`

## Dados de Treinamento

O dataset contém respostas ao questionário do ENADE. A variável `PC3_6` indica a evasão (1 para evasão, 0 caso contrário).

Após o pré-processamento:

- A variável `evadiu` foi criada com base em `PC3_6`
- 70% dos dados foram utilizados para treinamento
- 30% para teste

## Resultados do Modelo

Modelo: Random Forest

| Métrica    | Valor   |
|------------|---------|
| Acurácia   | 0.98    |
| Precisão   | 0.87    |
| Recall     | 0.68    |
| F1-Score   | 0.76    |

**Matriz de Confusão:**
```
[[9146   51]
 [ 161  345]]
```

## Considerações Éticas

- O modelo é baseado em dados auto-relatados e quantitativos.
- Não considera aspectos socioemocionais ou contextuais.
- Deve ser utilizado com responsabilidade para evitar reforço de vieses.

## Limitações e Recomendações

- A variável-alvo é desbalanceada (maioria dos alunos não evadiram).
- Sugere-se uso de técnicas como `SMOTE`, reamostragem ou ajuste de pesos.
- Validação cruzada pode melhorar a robustez dos resultados.

## Requisitos

- Python 3.x
- Bibliotecas: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `numpy`
