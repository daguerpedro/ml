[Seaborn](https://seaborn.pydata.org/api.html) é uma ferramenta para plotar gráficos em python baseada em [Matplotlib](https://matplotlib.org/stable/api/index).
## Importar
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
## Carregar dados [[Pandas]]
## Plotar
```python
serie = pd.read_csv('path/to/file.csv')
plt.figure(figsize=(14,6))
plt.title("Titulo Grafico")
sns.lineplot(data=serie)
```
## Plotar Séries específica
```python
plt.figure(figsize=(14,6))
plt.title("Titulo")
sns.lineplot(data=serie['Col A'], label="Serie A")
sns.lineplot(data=serie['Col B'], label="Serie B")
plt.xlabel("Titulo eixo X")
```