import pandas as pd

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('E:/Programação/Kaggle/datasets/home-data-for-ml-course/train.csv')
y = data['SalePrice']
data.drop(columns=['Id','SalePrice'], inplace=True)

num = data.select_dtypes(exclude=['object']).columns
cat = data.select_dtypes(include=['object']).columns

x_train, x_val, y_train, y_val = train_test_split(data, y, train_size=0.8, random_state=1)

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
    ),    
])

x_train_preproc = preproc.fit_transform(x_train)
x_val_preproc = preproc.transform(x_val)

model = XGBRegressor(
    random_state=1,    
    learning_rate=0.05,
    n_estimators=2000,
    subsample=0.75,    
    gamma=0.1,    
    max_depth=4,    
    eval_metric='mae',
    early_stopping_rounds=25,
    n_jobs=3
)               

model.fit(
    x_train_preproc,
    y_train,
    eval_set=[(x_val_preproc, y_val)],
    verbose=False,    
)
       
preds = model.predict(x_val_preproc)
mae = mean_absolute_error(y_true=y_val, y_pred=preds)

print(f"MAE: {mae:.3f}")