
# 🧠 Classificação Multiclasse com CNNs - CIFAR-10

Este projeto faz parte do **Trabalho Final (Parte 1)** da disciplina de Aprendizado de Máquina do PPgEEC.  
O objetivo foi desenvolver, testar e comparar diferentes arquiteturas de redes neurais convolucionais (CNNs) aplicadas à tarefa de **classificação multiclasse com o dataset CIFAR-10**.

---

## 📦 Estrutura do Projeto

```
TrabalhoFinal_Parte1/
├── data/                  # Dataset CIFAR-10
├── models/                # Arquiteturas de CNN
├── notebooks/             # Notebooks com experimentos
├── outputs/               # Gráficos e matrizes de confusão
│   ├── confusion_matrix/
│   └── graficos/
├── utils/                 # Funções auxiliares (train, test, métricas)
├── requirements.txt
└── README.md
```

---

## 📚 Modelos Testados

| Notebook | Modelo            | Descrição                                       |
|----------|-------------------|-------------------------------------------------|
| `01`     | Modelo Base       | CNN simples com 2 convoluções + FC             |
| `02`     | Modelo n_features | Aumento de filtros (nf1/nf2)                   |
| `03`     | Modelo Blocos     | CNN com blocos modulares (Conv+BN+ReLU+Pool)   |
| `04`     | Seu Modelo        | CNN personalizada com Dropout e BatchNorm      |
| `05`     | Análise Comparativa | Comparação gráfica entre os 4 modelos       |

---

## 🔎 Dataset

Utilizamos o **CIFAR-10**, que contém 60.000 imagens coloridas 32x32 distribuídas em 10 classes:

```
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

---

## 📊 Resultados por Modelo

### 📘 01_ModeloBase.ipynb

**📌 Matriz de Confusão**  
![Matriz Base](outputs/confusion_matrix/modelo_base_cm.png)

**📌 Gráfico Acurácia e Loss**  
![Gráfico Base](outputs/graficos/modelo_base_acc_loss.png)

---

### 📘 02_ModeloNFeatures.ipynb

**📌 Matriz de Confusão**  
![Matriz n_features](outputs/confusion_matrix/modelo_nfeatures_cm.png)

**📌 Gráfico Acurácia e Loss**  
![Gráfico n_features](outputs/graficos/modelo_nfeatures_acc_loss.png)

---

### 📘 03_ModeloBlocos.ipynb

**📌 Matriz de Confusão**  
![Matriz Blocos](outputs/confusion_matrix/modelo_blocos_cm.png)

**📌 Gráfico Acurácia e Loss**  
![Gráfico Blocos](outputs/graficos/modelo_blocos_acc_loss.png)

---

### 📘 04_SeuModelo.ipynb

**📌 Matriz de Confusão**  
![Matriz Seu Modelo](outputs/confusion_matrix/seu_modelo_cm.png)

**📌 Gráfico Acurácia e Loss**  
![Gráfico Seu Modelo](outputs/graficos/seu_modelo_acc_loss.png)

---

## 📈 Comparação Final (05_AnaliseComparativa.ipynb)

**🔁 Acurácia no Teste**  
![Comparativo Acurácia](outputs/graficos/comparativo_acc_teste.png)

**🔁 Loss no Teste**  
![Comparativo Loss](outputs/graficos/comparativo_loss_teste.png)

---

## 🚀 Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/TrabalhoFinal_Parte1.git
   cd TrabalhoFinal_Parte1
   ```

2. Instale os requisitos:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute os notebooks com o Jupyter ou dentro do VS Code.

---

## ✍️ Autor

- **Pietro Augusto de Albuquerque Lira e Silva**
- [pietrolira.com.br](https://pietrolira.com.br)
- Mestrando em Engenharia Elétrica e da Computação (PPgEEC/UFRN)

---

## 🧾 Licença

Este projeto é de uso acadêmico e educacional.
