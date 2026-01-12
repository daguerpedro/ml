Ao criar uma rede neural ([[Redes neurais]]) pela primeira vez, todos os pesos dos neurônios ([[Neurônio]]) são definidos aleatoriamente.
## Dados de treino
Os dados de treino consistem fornecer as entradas com os dados de saídas esperados pelo modelo.
## Treinar
Treinar uma rede neural significa ajustar os pesos de uma maneira que o modelo consiga transformar as entradas na saída esperada.
Para treinar um modelo é necessário:
- uma [[#Função objetivo (loss)]] que será otimizada através da avaliação da qualidade do modelo. Também conhecida como função de perda (`loss`).
- uma Função de otimização ([[Stochastic Gradient Descent]]) para indicar ao modelo como alterar os pesos.
## Função objetivo (loss)
De maneira similar ao processo de [[Validação]], a função de perda mede a disparidade entre o resultado esperado e o valor determinado pelo modelo. Problemas diferentes terão funções objetivos diferentes.

Algumas funções objetivo comuns são:
- **Erro Absoluto Médio** (``MAE``)
- **Erro Quadrático Médio** (``MSE``)
- **Perda de Huber** (``Huber loss``)

## Compilação
Após definir um modelo, você pode adicionar uma função de perda e um otimizador com o método compile. 
Por exemplo, usando o modelo [[Stochastic Gradient Descent#Adam]]
```python
model.compile(
    optimizer="adam",
    loss="mae",
)
```