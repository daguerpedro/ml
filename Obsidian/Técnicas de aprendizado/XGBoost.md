### Princípio
O modelo de boost de gradiente é uma alternativa aos modelos de [[Árvore de decisão]] e [[Random Forest]], que utiliza o método do gradiente (Gradient Descent) para otimizar a função de "perda" (Como por exemplo a função do MAE [[Validação]]).
## Utilização
A biblioteca [xgboost](https://xgboost.readthedocs.io/en/latest/index.html) possui modelos melhor otimizados em relação aos modelos do [[Scikit-learn]]
Parâmetros importantes que alteram drasticamente a velocidade e precisão do treinamento, tais como:
- ``n_estimators``: 
	specifies how many times to go through the modeling cycle described above. It is equal to the number of models that we include in the ensemble.
	```python
	my_model = XGBRegressor(n_estimators=500)
	my_model.fit(X_train, y_train)
	```
- ``early_stopping_rounds``:
	offers a way to automatically find the ideal value for `n_estimators`. Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for `n_estimators`. It's smart to set a high value for `n_estimators` and then use `early_stopping_rounds` to find the optimal time to stop iterating.
	```python
	my_model = XGBRegressor(n_estimators=500)
	my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)],
             verbose=False)
	```
- ``learning_rate``:
	Instead of getting predictions by simply adding up the predictions from each component model, we can multiply the predictions from each model by a small number (known as the **learning rate**) before adding them in.
	```python
	my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
	my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
	```
- ``n_jobs``:
	On larger datasets where runtime is a consideration, you can use parallelism to build your models faster. It's common to set the parameter `n_jobs` equal to the number of cores on your machine. On smaller datasets, this won't help.
	```python
	my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
	my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
	```
### Código:
```python
from xgboost import XGBRegressor

model = XGBRegressor()
model.fit(X, y)
```