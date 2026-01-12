Para modelos de regressão é necessário possuir dados numéricos.
As alternativas são remover os dados categóricos ou transforma-los em numéricos.
```python
# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)
```
### Remover dados categóricos
```python
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
```
### Ordinal encoding
Essa abordagem assume uma ordem de classificação numérica, por exemplo: 

| Café da manhã       | Encoding |
| ------------------- | -------- |
| Todo dia            | 3        |
| Nunca               | 0        |
| Raramente           | 1        |
| Na maioria dos dias | 2        |
| Nunca               | 0        |
Onde a ordem é "Nunca" (0) < "Raramente" (1) < "Na maioria dos dias" (2) < "Todo dia" (3)
```python
from sklearn.preprocessing import OrdinalEncoder

# Make copy to avoid changing original data 
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])
```
### One-Hot encoding
Essa abordagem cria novas colunas que indicam a presença ou ausência de cada possível valor do dado original

| Cor      |     | Vermelho | Amarelo | Verde |
| -------- | --- | -------- | ------- | ----- |
| Vermelho |     | 1        | 0       | 0     |
| Vermelho |     | 1        | 0       | 0     |
| Amarelo  |     | 0        | 1       | 0     |
| Verde    |     | 0        | 0       | 1     |
```python
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Ensure all columns have string type
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)
```