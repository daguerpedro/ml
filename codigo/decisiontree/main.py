from sklearn import tree
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score

import pandas as pd

# Carrega o CSV com os dados de venda
data = pd.read_csv('E:/Programação/Kaggle/datasets/home-data-for-ml-course/train.csv')
data = data.drop(['Id'], axis=1)

# Separa os valores de vendas pois queremos fazer predições
raw_x = data.drop(['SalePrice'], axis=1)
raw_y = data['SalePrice']

# Separa colunas numericas e categoricas para aplicar transformações
numeric_cols   = raw_x.select_dtypes(exclude=['object']).columns
categoric_cols = raw_x.select_dtypes(include=['object']).columns

# Define as estratégias para valores numéricos nulos
numeric_imputer = SimpleImputer(strategy='median')

# Define as estratégias para valores categóricos
categoric_imputer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))    
])

# Camada de preprocessamento da pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_imputer, numeric_cols),
        ('cat', categoric_imputer, categoric_cols)
    ]
)

# Definição do modelo com valores encontrados
model = tree.DecisionTreeRegressor(
    random_state=10,
    max_depth=11,
    min_samples_leaf=2
)
    
pipe = Pipeline(steps=[('preprocessor', preprocessor),('model', model)])
scores = cross_val_score(
    pipe,
    X=raw_x,
    y=raw_y,
    scoring='neg_mean_absolute_error',
    cv=5
)
maes = -1*scores

print("MAE por fold:", maes)
print("MAE médio:", maes.mean())
print("Desvio padrão:", maes.std())