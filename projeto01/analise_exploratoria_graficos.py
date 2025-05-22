# analise_exploratoria_graficos.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar dados
caminho_arquivo = 'enape_raw_dataset.csv'
df = pd.read_csv(caminho_arquivo)

# Exibir as primeiras linhas e tipos
print("\n🔹 Primeiras linhas:")
print(df.head())

print("\n🔹 Informações gerais:")
df.info()

print("\n🔹 Valores ausentes:")
print(df.isnull().sum())

print("\n🔹 Estatísticas descritivas (numéricas):")
print(df.describe())

# Estatísticas descritivas categóricas
if any(df.dtypes == 'object'):
    print("\n🔹 Estatísticas descritivas (categóricas):")
    print(df.describe(include='object'))

# -------- GRÁFICOS --------
sns.set(style="whitegrid")

# 1. Distribuição de variáveis numéricas (exemplo com idade, se existir)
if 'idade' in df.columns:
    df['idade'].hist(bins=30)
    plt.title('Distribuição da Idade')
    plt.xlabel('Idade')
    plt.ylabel('Frequência')
    plt.show()

# 2. Gráfico de barras da variável PC3_6 (indicador de evasão)
if 'PC3_6' in df.columns:
    sns.countplot(data=df, x='PC3_6')
    plt.title('Distribuição da Evasão (PC3_6)')
    plt.xlabel('Evadido (1 = Sim, 0 = Não)')
    plt.ylabel('Contagem')
    plt.show()

# 3. Boxplots de variáveis numéricas por evasão
if 'PC3_6' in df.columns:
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if col != 'PC3_6':
            sns.boxplot(x='PC3_6', y=col, data=df)
            plt.title(f'{col} por Classe de Evasão')
            plt.show()

# 4. Gráficos de barras para variáveis categóricas
for col in df.select_dtypes(include='object').columns:
    if df[col].nunique() < 20:
        sns.countplot(data=df, x=col, hue='PC3_6') if 'PC3_6' in df.columns else sns.countplot(data=df, x=col)
        plt.title(f'Distribuição da variável {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# 5. Mapa de calor da correlação
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()

# 6. Gráfico de dispersão exemplo (idade vs outra variável)
if 'idade' in df.columns and 'nota_final' in df.columns:
    sns.scatterplot(data=df, x='idade', y='nota_final', hue='PC3_6' if 'PC3_6' in df.columns else None)
    plt.title('Idade vs Nota Final')
    plt.xlabel('Idade')
    plt.ylabel('Nota Final')
    plt.show()

print("\n✅ Análise exploratória finalizada com gráficos.")
