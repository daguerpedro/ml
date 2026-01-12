Feature Importance é o conceito de quais variáveis tem maior impacto nas predições de um modelo.
## Permutation Importance
Permutation Importance é uma maneira rápida para calcular o peso das variáveis de um modelo.
O método consiste em:
1. Treinar um modelo.
2. Fazer predições embaralhando as variáveis de uma única coluna dos dados de treino.
3. Desembaralhar a coluna e repetir a etapa ``2.`` com a próxima coluna.
## Análise de resultado
```python
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
```
Os valores mais ao topo da lista são as variáveis de maior importância.
O primeiro valor de cada linha representa quanto a performance do modelo mudou com o embaralhamento de cada variável (Nesse caso, usando "acurácia" como métrica de performance).
