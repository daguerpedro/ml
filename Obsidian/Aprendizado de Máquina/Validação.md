Para avaliar a qualidade dos modelos deve-se utilizar funções objetivo com métricas de significância estatística, tais como:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSe)
Por exemplo:
```python
from sklearn.metrics import mean_absolute_error

predicted = model.predict(X)
mean_absolute_error(y, predicted)
```

É importante avaliar e comparar os pesos de diversos parâmetros do modelo, como por exemplo [[Árvore de decisão]]:

```python
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
    
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
    
    # Max leaf nodes: 5  		 Mean Absolute Error:  347380
	# Max leaf nodes: 50  		 Mean Absolute Error:  258171
	# Max leaf nodes: 500  		 Mean Absolute Error:  243495
	# Max leaf nodes: 5000  		 Mean Absolute Error:  254983
```
Outra alternativa é usar modelos mais sofisticados como [[Random Forest]].