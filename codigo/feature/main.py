import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from xgboost import XGBRegressor

data = pd.read_csv('E:/Programação/Kaggle/datasets/home-data-for-ml-course/train.csv')
data.drop(columns=['Id'], inplace=True)
y=data.pop('SalePrice')

#Transformação de variáveis
#data['LotFrontageRatio'] = data.LotFrontage/data.LotArea
data['Residential'] = data.MSZoning.isin(['FV', 'RH', 'RL', 'RP', 'RM'])
data['RegularShape'] = data.LotShape == 'Reg'
data['LastRemod'] = data.YearRemodAdd - data.YearBuilt

data['HasBasement'] = (data['TotalBsmtSF'] > 0).astype(int)
data['BsmtFinRatio'] = np.where(
    data['TotalBsmtSF'] > 0,
    (data['BsmtFinSF1'] + data['BsmtFinSF2']) / data['TotalBsmtSF'],
    0.0
)
data['BsmtUnfRatio'] = np.where(
    data['TotalBsmtSF'] > 0,
    data['BsmtUnfSF'] / data['TotalBsmtSF'],
    0.0
)

data['FlrSF'] = data['1stFlrSF'] + data['2ndFlrSF']
data['Bath'] = data.BsmtFullBath + data.BsmtHalfBath/2 + data.FullBath + data.HalfBath/2

data['Age'] = data.YrSold - data.YearBuilt
data['RemodAge'] = data.YrSold - data.YearRemodAdd
data['TotalSF'] = data.GrLivArea + data.TotalBsmtSF

data.drop(columns=[
    'BsmtFinSF1',
    'BsmtFinSF2',
    'BsmtUnfSF',
    '1stFlrSF',
    '2ndFlrSF',
    'BsmtFullBath',
    'BsmtHalfBath',
    'FullBath',
    'HalfBath',
], inplace=True)

num = data.select_dtypes(exclude=['object']).columns
cat = data.select_dtypes(include=['object']).columns

preproc = ColumnTransformer(
    transformers=[
        ('num',
         Pipeline(steps=[
             ('imputer', SimpleImputer(strategy='median', add_indicator=True))
         ]),
         num),

        ('cat',
         Pipeline(steps=[
             ('imputer', SimpleImputer(strategy='most_frequent')),
             ('encoder', OneHotEncoder(handle_unknown='ignore',sparse_output=False))
         ]),
         cat)
    ],
    remainder='drop'
)

x_train, x_val, y_train, y_val = train_test_split(data, y, train_size=0.8, random_state=1)

x_train_preproc = preproc.fit_transform(x_train)
x_val_preproc = preproc.transform(x_val)

model = XGBRegressor(
    random_state=1,    
    learning_rate=0.05,
    n_estimators=2000,
    subsample=0.75,    
    gamma=0.1,    
    max_depth=6,    
    eval_metric='mae',
    n_jobs=3,
    min_child_weight=7,
    colsample_bytree=0.8,
    early_stopping_rounds=25,
)     
          
model.fit(
    x_train_preproc,
    y_train,
    eval_set=[(x_val_preproc, y_val)],
    verbose=False
)

preds = model.predict(x_val_preproc)
mae = mean_absolute_error(y_true=y_val, y_pred=preds)

print(f"MAE: {mae:.3f}")