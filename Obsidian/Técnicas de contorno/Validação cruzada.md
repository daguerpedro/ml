A técnica de validação cruzada consiste em dividir os dados do modelo em diferentes partes e usa-los para [[Treino]] e [[Validação]], avaliando quão bem o modelo generaliza dados ainda não vistos.  Por exemplo, dividimos os dados em 5 partes e para cada parte utilizamos os dados como validação e as outras partes como treino.

Validação cruzada devolve uma métrica de qualidade de modelo mais precisa, no entanto pode levar mais tempo para ser avaliada.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50,
                                                              random_state=0))
                             ])
                             
                             
from sklearn.model_selection import cross_val_score

# We set the number of folds with the `cv` parameter.
# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)
```