Redes neurais tipicamente é a organização neurônios ([[Neurônio]]) em [[#Camadas]].
## Camadas
Cada camada pode ser interpretada como uma transformação simples.
Com várias camadas "empilhadas", uma rede neural pode transformar as variáveis de entrada em maneiras mais complexas. Em uma rede neural bem treinada, cada camada é uma transformação se aproximando para uma solução.
## Funções de ativação
Funções de ativação são responsáveis pelas relações não lineares em redes neurais. 
Uma função de ativação pode ser definida como uma função que aplicamos na saída de cada camada.  
## Retificadora
A função de ativação mais comum é a função retificadora `max(0, x)`.
A função retificadora é responsável por retificar a parte negativa para zero.
## Unidade Linear Retificada
Quando retificamos uma entrada linear temos uma Unidade Linear Retificada (**rectified linear unit** ou **ReLU**).
Ao aplicar uma ReLU para uma unidade linear a saída se torna 
$$y=max(0,w*x+b)$$
## Empilhando camadas densas
Ao adicionar não linearidades às entradas com ReLUs é possível obter transformações de dados mais complexos. No entanto, a saída final da camada sempre será uma unidade linear, tornando a rede apropriada para regressões.
As camadas antes da saída são chamadas de camadas **escondidas** (`hidden`), uma vez que não veremos a saída diretamente.
## Modelos sequenciais
O modelo `Sequential` ([[Neurônio#Neurônio no Keras]]) irá conectar uma lista de camadas na ordem da primeira para a última. Onde a primeira camada recebe a entrada e a última camada produz a saída.
```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),
    # the linear output layer 
    layers.Dense(units=1),
])
```
As camadas devem ser passadas em uma lista do tipo `[layer, layer, layer, ...]`.  Para adicionar uma função de ativação em uma camada, adicione o nome da função no argumento `activation`.