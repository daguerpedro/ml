As alternativas de contorno para valores nulos são:
- Remover as entradas utilizando [[Pandas#Remover nulos]]
- Remover as colunas:
	```python
	cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
	# Drop columns in training and validation data
	reduced_X_train = X_train.drop(cols_with_missing, axis=1)
	reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)
	```
- Mas a mais recomendada é: [[Imputation]]