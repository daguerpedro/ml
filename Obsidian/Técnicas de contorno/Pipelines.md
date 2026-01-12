Pipelines são maneiras de organizar e modularizar o pré-processamento de dados.
1. Definir as etapas da pipeline.
2. Definir o modelo.
3. Criar e avaliar a pipeline.
## Utilização
[Docs](https://scikit-learn.org/stable/api/sklearn.pipeline.html)
Exemplo de pipeline:
1. Definir etapas
	1. Contornar [[Valores nulos]] com [[Imputation]]
	2. Contornar [[Valores categóricos]] com [[Valores categóricos#One-Hot encoding]]

	```python
	from sklearn.compose import ColumnTransformer
	from sklearn.pipeline import Pipeline
	from sklearn.impute import SimpleImputer
	from sklearn.preprocessing import OneHotEncoder
	
	# Preprocessing for numerical data
	numerical_transformer = SimpleImputer(strategy='constant')
	
	# Preprocessing for categorical data
	categorical_transformer = Pipeline(steps=[
	    ('imputer', SimpleImputer(strategy='most_frequent')),
	    ('onehot', OneHotEncoder(handle_unknown='ignore'))
	])
	
	# Bundle preprocessing for numerical and categorical data
	preprocessor = ColumnTransformer(
	    transformers=[
	        ('num', numerical_transformer, numerical_cols),
	        ('cat', categorical_transformer, categorical_cols)
	    ])
	```

2. Definir o modelo:
	Utilizando um modelo de [[Random Forest]]
	```python
	from sklearn.ensemble import RandomForestRegressor
	model = RandomForestRegressor(n_estimators=100, random_state=0)
	```
3. Criar e avaliar a pipeline:
	Após a predição basta passar pelo processo de [[Treino]] e [[Validação]]
	```python
	from sklearn.metrics import mean_absolute_error
	
	# Bundle preprocessing and modeling code in a pipeline
	my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),('model', model)])
	
	# Preprocessing of training data, fit model 
	my_pipeline.fit(X_train, y_train)
	
	# Preprocessing of validation data, get predictions
	preds = my_pipeline.predict(X_valid)
	
	# Evaluate the model
	score = mean_absolute_error(y_valid, preds)
	```