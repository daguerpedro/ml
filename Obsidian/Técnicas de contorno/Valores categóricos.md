Para modelos de regressão é necessário possuir dados numéricos.
As alternativas são remover os dados categóricos ou transforma-los em numéricos.
### Remover dados categóricos
```python
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
```
### Ordinal encoding
Essa abordagem assume uma ordem de classificação numérica, por exemplo: 
É mais eficiente/limpo utilizar dentro de uma [[Pipelines]] no pré-processamento com [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)

| Café da manhã       | Encoding |
| ------------------- | -------- |
| Todo dia            | 3        |
| Nunca               | 0        |
| Raramente           | 1        |
| Na maioria dos dias | 2        |
| Nunca               | 0        |
Onde a ordem é "Nunca" (0) < "Raramente" (1) < "Na maioria dos dias" (2) < "Todo dia" (3)
### One-Hot encoding
Essa abordagem cria novas colunas que indicam a presença ou ausência de cada possível valor do dado original
É mais eficiente/limpo utilizar dentro de uma [[Pipelines]] no pré-processamento com [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)

| Cor      |     | Vermelho | Amarelo | Verde |
| -------- | --- | -------- | ------- | ----- |
| Vermelho |     | 1        | 0       | 0     |
| Vermelho |     | 1        | 0       | 0     |
| Amarelo  |     | 0        | 1       | 0     |
| Verde    |     | 0        | 0       | 1     |
