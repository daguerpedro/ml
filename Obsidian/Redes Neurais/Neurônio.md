Um neurônio é uma unidade de uma rede neural responsável por uma única computação.
## Entradas
O neurônio é um modelo de uma entrada linear do tipo:
$$y = w*x + b$$
Onde ***w*** é o peso, ***x*** é a entrada e ***b*** é o bias.
As redes neurais "aprendem" ao modificar os pesos ***w*** de cada neurônio.

Um único neurônio também pode assumir múltiplas variáveis de entrada, assumindo a forma
$$y = w_0*x_0 + ... + w_n*x_n + b $$
## Neurônio no Keras
- `units` é o número de saídas
- `input_shape` é o número de variáveis
```python
from tensorflow import keras
from tensorflow.keras import layers

# Create a network with 1 linear unit
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])
```