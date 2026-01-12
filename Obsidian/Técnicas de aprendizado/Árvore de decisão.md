### Princípio
Uma árvore de decisão é um modelo de regressão que segue o princípio de um fluxograma:
1. Quebrar dados em grupos
2. Capturar padrões
3. Predição baseada em fit/treino
## Utilização
[[Scikit-learn#Regressão]]
Parâmetros importantes:
- ``max_depth``: int
- ``max_leaf_nodes``: int
- ``random_state``: int
```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=1) 
model.fit(X, y)
model.predict(X)
```