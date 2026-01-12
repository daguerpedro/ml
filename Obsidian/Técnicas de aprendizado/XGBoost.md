### Princípio
O modelo de boost de gradiente é uma alternativa aos modelos de [[Árvore de decisão]] e [[Random Forest]], que utiliza o método do gradiente (Gradient Descent) para otimizar a função de "perda" (Como por exemplo a função do MAE [[Validação]]).
## Utilização
A biblioteca [xgboost](https://xgboost.readthedocs.io/en/latest/index.html) possui modelos melhor otimizados e implementações próprias de [[Scikit-learn]] | [Scikit-learn Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html).
Parâmetros importantes:
- ``n_estimators``: 
	Numero de modelos/árvores de boost de gradiente.
- ``early_stopping_rounds``:
	Método para encontrar um numero ideal para `n_estimators`. Faz com que o modelo pare de iterar quando o score de validação parara de melhorar, mesmo que não seja o melhor ponto global. **É interessante escolher um n_estimators alto e usar o early_stopping_rounds para achar o valor otimizado**.
- ``learning_rate``:
	Instead of getting predictions by simply adding up the predictions from each component model, we can multiply the predictions from each model by a small number (known as the **learning rate**) before adding them in.
- ``n_jobs``: threads.
### Código:
```python
from xgboost import XGBRegressor

model = XGBRegressor()
model.fit(X, y)
```