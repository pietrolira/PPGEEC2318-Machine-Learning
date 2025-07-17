# PPGEEC2318-Machine-Learning
Projetos da disciplina Aprendizado de máquina do mestrado PPGEEC2318/UFRN


# Classificação Multiclasse com CNNs - CIFAR-10

Este projeto faz parte do **Trabalho Final (Parte 1)** da disciplina de Aprendizado de Máquina do PPgEEC.  
O objetivo foi desenvolver, testar e comparar diferentes arquiteturas de redes neurais convolucionais (CNNs) aplicadas à tarefa de **classificação multiclasse com o dataset CIFAR-10**.

---

## Estrutura do Projeto

```
TrabalhoFinal_Parte1/
├── models/                # Arquiteturas de CNN
├── notebooks/             # Notebooks com experimentos
├── utils/                 # Funções auxiliares (train, test, métricas)
├── requirements.txt
└── README.md
```

---

## Modelos Testados

| Notebook | Modelo            | Descrição                                       |
|----------|-------------------|-------------------------------------------------|
| `01`     | Modelo Base       | CNN simples com 2 convoluções + FC             |
| `02`     | Modelo n_features | Aumento de filtros (nf1/nf2)                   |
| `03`     | Modelo Blocos     | CNN com blocos modulares (Conv+BN+ReLU+Pool)   |
| `04`     | Seu Modelo        | CNN personalizada com Dropout e BatchNorm      |
| `05`     | Análise Comparativa | Comparação gráfica entre os 4 modelos       |

---

## Dataset

Utilizamos o **CIFAR-10**, que contém 60.000 imagens coloridas 32x32 distribuídas em 10 classes:

Não é preciso baixar manualmente o cifar-10-batches-py/
isto ocorre no próprio código em PyTorch que faz isso automaticamente ao rodar os notebooks.

Em cada um dos arquivos existe a célula abaixo:

```python
from torchvision import datasets, transforms

transform = transforms.ToTensor()

# Faz o download automático do CIFAR-10 e extrai na pasta 'data/'
datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

```
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

---

## Resultados por Modelo

### 01_ModeloBase.ipynb

**Matriz de Confusão**  
![Matriz Base](graficos/1matriz_de_confusao.png)

**Gráfico Acurácia e Loss**  
![Gráfico Base](graficos/1graficos.png)

---

### 02_ModeloNFeatures.ipynb

**Matriz de Confusão**  
![Matriz n_features](graficos/2matriz_de_confusao.png)

**Gráfico Acurácia e Loss**  
![Gráfico n_features](graficos/2graficos.png)

---

### 03_ModeloBlocos.ipynb

**Matriz de Confusão**  
![Matriz Blocos](graficos/3matriz_de_confusao.png)

**Gráfico Acurácia e Loss**  
![Gráfico Blocos](graficos/3graficos.png)

---

### 04_SeuModelo.ipynb

**Matriz de Confusão**  
![Matriz Seu Modelo](graficos/4matriz_de_confusao.png)

**Gráfico Acurácia e Loss**  
![Gráfico Seu Modelo](graficos/4graficos.png)

---

## Comparação Final (05_AnaliseComparativa.ipynb)

**Acurácia no Teste**  
![Comparativo Acurácia](graficos/5_grafico_comparativo_1.png)

**Loss no Teste**  
![Comparativo Loss](graficos/5_grafico_comparativo_2.png)

---

## Autor

- **Pietro Augusto de Albuquerque Lira e Silva**
- [pietrolira.com.br](https://pietrolira.com.br)
- Mestrando em Engenharia Elétrica e da Computação (PPgEEC/UFRN)

---
