### Princípio
Random forest é um modelo que combina diversas [[Árvore de decisão]] reduzindo overfitting.
## Utilização
[[Scikit-learn]] | [Docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
Parâmetros importantes:
- `n_estimators`: int
- ``max_depth``: int
- ``max_leaf_nodes``: int
- ``random_state``: int
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=1)
model.fit(X, y)
model.predict(X)
```