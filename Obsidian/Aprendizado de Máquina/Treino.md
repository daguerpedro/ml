Antes do processo de [[Validação]] de modelos, é necessário reservar uma parte dos dados de treino para serem usados exclusivamente como validação. 

Ao usar somente dados de treino para validar, a métrica resultante não refletirá um resultado real da qualidade do modelo quando utilizados novos dados, resultando em [[#Overfitting]] ou [[#Underfitting]].

Exemplo de treino para um modelo de [[Árvore de decisão]]:

```python
from sklearn.model_selection import train_test_split
# Separar parte para treinar e parte para validar
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
model = DecisionTreeRegressor()
# Regredir usando dados para treino
model.fit(train_X, train_y) 
# Validar usando dados de validação
val_predictions = model.predict(val_X)
# Métrica de qualidade MAE
mean_absolute_error(val_y, val_predictions)
```
## Overfitting 
Quando o modelo tem uma boa predição para dados de treino mas não consegue prever com precisão dados de validação.
## Underfitting
Quando o modelo é muito raso e não alcança boa precisão para dados de treino.
