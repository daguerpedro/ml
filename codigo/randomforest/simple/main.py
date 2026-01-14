import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

data = pd.read_csv('E:/Programação/Kaggle/datasets/home-data-for-ml-course/train.csv')
data.drop(columns=['Id'], inplace=True)

x = data.drop(columns=['SalePrice'], axis=1)

cat = x.select_dtypes(include=['object']).columns
num = x.select_dtypes(exclude=['object']).columns

preproc = ColumnTransformer(transformers=[
    (
        'num', 
        Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median', add_indicator=True))
        ]), 
        num
    ),
    (
        'cat', 
        Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing', add_indicator=True)),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), 
        cat
    )
])

pipe = Pipeline(steps=[
    ('preprocess', preproc),
    ('model', RandomForestRegressor(random_state=1))
])

maes = -1*cross_val_score(
    pipe, 
    X=x, 
    y=data['SalePrice'],
    scoring='neg_mean_absolute_error',
    cv=5,
    n_jobs=-1
)

print(f"Min: {round(maes.min(), 4)}\nAvg; {round(maes.mean(), 4)}\nStd: {round(maes.std(), 4)}")