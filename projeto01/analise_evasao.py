import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Carregar os dados
def carregar_dados(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo)
    return df

# 2. Criar variável alvo com base na coluna PC3_6
def criar_variavel_alvo(df):
    # Exemplo: se 'PC3_6' == 1 significa que evadiu, ajuste conforme seu dicionário
    df['evadiu'] = df['PC3_6'].apply(lambda x: 1 if x == 1 else 0)
    return df

# 3. Pré-processamento dos dados
def preprocessar_dados(df):
    # Remover a coluna PC3_6, pois agora usamos 'evadiu' como alvo
    df = df.drop(columns=['PC3_6'])

    # Preencher valores nulos (melhor que dropna)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    # Transformar variáveis categóricas
    df = pd.get_dummies(df, drop_first=True)

    return df

# 4. Treinamento do modelo
def treinar_modelo(X_train, y_train):
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    return modelo

# 5. Avaliação do modelo
def avaliar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))
    print("\nRelatório de classificação:\n", classification_report(y_test, y_pred))

# 6. Função principal
def main():
    df = carregar_dados('enape_raw_dataset.csv')

    df = criar_variavel_alvo(df)
    df = preprocessar_dados(df)

    X = df.drop('evadiu', axis=1)
    y = df['evadiu']

    if len(df) < 2:
        print("Erro: Dataset com poucas linhas após o pré-processamento.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    modelo = treinar_modelo(X_train, y_train)
    avaliar_modelo(modelo, X_test, y_test)

if __name__ == '__main__':
    main()
