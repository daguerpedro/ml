Imputation é a técnica de contorno para [[Valores nulos]] de inserção de dados.
Substitui dados nulos usando estatística descritiva (como média, mediana, menor ou mais frequente) ao longo de cada coluna ou usando uma constante.
[[Scikit-learn]]
## Utilização
[Docs](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
É mais eficiente/limpo utilizar dentro de uma [[Pipelines]] no pré-processamento com [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html).
Parâmetros:
- ``missing_values``: **int, float, str, np.nan, None or pandas.NA, default=np.nan**
	O placeholder para valores nulos, ao usar [[Pandas]] pode assumir `np.nan` ou `np.NA`
- ``strategy``: **str or Callable, default=’mean’**
	- If “mean”, then replace missing values using the mean along each column. Can only be used with numeric data.
	- If “median”, then replace missing values using the median along each column. Can only be used with numeric data.
	- If “most_frequent”, then replace missing using the most frequent value along each column. Can be used with strings or numeric data. If there is more than one such value, only the smallest is returned.
	- If “constant”, then replace missing values with fill_value. Can be used with strings or numeric data.
	- If an instance of Callable, then replace missing values using the scalar statistic returned by running the callable over a dense 1d array containing non-missing values of each column.
- ``add_indicator``: bool
	If True, a [`MissingIndicator`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html#sklearn.impute.MissingIndicator "sklearn.impute.MissingIndicator") transform will stack onto output of the imputer’s transform. This allows a predictive estimator to account for missingness despite imputation. If a feature has no missing values at fit/train time, the feature won’t appear on the missing indicator even if there are missing values at transform/test time.