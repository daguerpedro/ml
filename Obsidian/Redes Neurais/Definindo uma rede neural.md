Para montar uma [[Redes neurais]] é preciso definir alguns aspectos, como:
- Dados de treino [[Aprendizado neural#Dados de treino]]
- Número de entradas [[Neurônio#Entradas]] da rede
- Função de perda [[Aprendizado neural#Função objetivo]]
- Função de otimização [[Stochastic Gradient Descent#Adam]]
- Definir um tamanho de [[Stochastic Gradient Descent#Batch e Epoch]]
# Modelo
para esse exemplo, seguiremos o modelo Kaggle [Example - Red Wine Quality](https://www.kaggle.com/code/ryanholbrook/stochastic-gradient-descent?scriptVersionId=126574203&cellId=2)
## Número de entradas
Para definir o número de entradas (``inputs``) podemos analisar o número de colunas presentes nos dados de treino. 

Por exemplo, 11 colunas significam 11 entradas:
```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=[11]),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])
```
## Otimizador e função de perda
Após definir o modelo, precisamos compila-lo com as funções de otimização e perda.
Exemplo:
```python
model.compile(
    optimizer='adam',
    loss='mae',
)
```
## Treino
Assim é possível começar o treinamento.
Definimos ao Keras para alimentar o otimizador com 256 linhas dos dados de treino por vez (**Batch size**) e fazer isso 10 vezes até o final dos dados de treino (**epochs**).

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=10,
)
```