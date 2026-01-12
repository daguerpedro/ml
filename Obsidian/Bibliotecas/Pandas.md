[Pandas](https://pandas.pydata.org/docs/reference/index.html) é uma biblioteca para análise e manipulação de dados.

```python
import pandas as pd
```
## Series
Uma ndarray [(numpy)](https://numpy.org/doc/stable/reference/index.html#reference) unidimensional com rótulos no eixo.
Como uma lista.
```python
pd.Series(['A', 'B', 'C'])
```
## DataFrame
Dados tabulares bidimensionais de tamanho mutável. 
Como uma tabela.
```python
pd.DataFrame({'Produto': ['A', 'B', 'C'], 'Preço': [23, 22, 33]})
```
## Carregar dados
```python
pd.read_csv('path/to/file.csv')
pd.read_csv('path/to/file.csv', index_col=0)
```
## Leitura resumida
```python
serie.head() # Primeiros dados da tabela/lista
serie.tail() # Últimos dados tabela/lista
```
## Busca
Por indexação:
```python
serie.iloc[0] # No "ID" 0
serie.iloc[row, col] 
```
Por texto ou condicional (é possível usar [[Lista#Comprehensions]]):
```python
serie.loc[0, 'country'] 
serie.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]
serie.loc[serie.country == 'Italy']
serie.loc[serie.country.isin(['Italy', 'France'])]
```
## Descrição de dados
```python
serie.describe()
serie.mean()
```
## Remover nulos
```python
serie.dropna()
```