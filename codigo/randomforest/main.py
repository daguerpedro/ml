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

x_train, x_val, y_train, y_val = train_test_split(data.drop(columns=['SalePrice'], axis=1), data['SalePrice'], train_size=0.8, random_state=1)

cat = x_train.select_dtypes(include=['object']).columns
num = x_train.select_dtypes(exclude=['object']).columns

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

pipe.fit(x_train, y_train)
preds = pipe.predict(x_val)
mae = mean_absolute_error(y_true=y_val, y_pred=preds)

print(mae)