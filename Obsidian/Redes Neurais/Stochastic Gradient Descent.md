Dentro do processo de [[Aprendizado neural]], a função de otimização é um algoritmo responsável por ajustar os pesos do modelo visando minimizar as perdas (``loss``).

No processo de **Deep Learning** todos os algoritmos de otimização usam o método do gradiente estocástico (**Stochastic Gradient Descent** ou **SGD**).
## Algoritmo 
O método do gradiente estocástico é um algoritmo iterativo que treina as [[Redes neurais]] nos seguintes passos:
1. Amostrar dados de treino ([[Aprendizado neural#Dados de treino]]) e utilizá-los na rede neural para fazer predições.
2. Medir a perda (`loss`) entre as predições e os valores esperados.
3. Ajustar os pesos em uma direção que torne a perda menor.
O algoritmo é repetido até que a perda seja tão pequena quanto desejado ou até que o algoritmo pare de fazer progresso.
## Batch e Epoch
A amostra de dados de cada iteração é chamada de **minibatch** ou **batch**, enquanto a rodada completa de dados é chamada de **epoch**. O número de epochs que você treina é de acordo com quantas vezes a rede verá cada exemplo de treino.
## Learning rate e Batch size
A taxa de aprendizado (**learning rate**) é o que limita o progresso do algoritmo em cada iteração.
Uma taxa de aprendizado menor significa que a rede precisa ver mais **minibatches** antes dos pesos convergirem para o melhor valor.

A taxa de aprendizado e o tamanho dos batches são os dois parâmetros que mais tem efeito em no processo de treino por **SGD**. 
## Adam
**Adam** é um algoritmo **SGD** que possui uma taxa de aprendizado adaptativa o que o torna viável para a maioria dos problemas sem precisar de refinamento de parâmetro **(parameter tuning**).