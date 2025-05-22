#------------- PARTE 01 - CARREGAMENTO E EXPLORAÇÃO INICIAL ----------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
df = pd.read_csv('enape_raw_dataset.csv')

# Visualizar as primeiras linhas
print("\n🔹 Primeiras linhas do dataset:")
print(df.head())

# Lista de colunas e tipos de dados
print("\n🔹 Informações do dataset:")
df.info()

# Total de valores ausentes por coluna
print("\n🔹 Valores ausentes (total):")
print(df.isnull().sum())

# Em percentual:
print("\n🔹 Valores ausentes (percentual):")
print((df.isnull().sum() / len(df)) * 100)

# Estatísticas para variáveis numéricas
print("\n🔹 Estatísticas - variáveis numéricas:")
print(df.describe())

# Estatísticas para variáveis categóricas, se houver
if any(df.dtypes == 'object'):
    print("\n🔹 Estatísticas - variáveis categóricas:")
    print(df.describe(include=['object']))
else:
    print("\nℹ️ Nenhuma variável categórica (tipo object) encontrada.")

# Distribuição da variável alvo 'evaded', se existir
if 'evaded' in df.columns:
    print("\n🔹 Distribuição da variável 'evaded':")
    print(df['evaded'].value_counts(normalize=True) * 100)

#------------- PARTE 02 - VISUALIZAÇÕES ----------------

# Histograma da idade, se existir
if 'idade' in df.columns:
    df['idade'].hist(bins=30)
    plt.title('Distribuição de Idade')
    plt.xlabel('Idade')
    plt.ylabel('Frequência')
