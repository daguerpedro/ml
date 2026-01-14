from re import search
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import cross_val_score

#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

data = pd.read_csv('E:/Programação/Kaggle/datasets/home-data-for-ml-course/train.csv')
data.drop(columns=['Id'], inplace=True)

x = data.drop(columns=['SalePrice'], axis=1)
y = data['SalePrice']

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

params = {
    "model__max_depth": [None, 10, 20, 30],
    "model__n_estimators": randint(100, 600),
    "model__max_features": uniform(0.3, 0.7),
    "model__min_samples_split": randint(2, 20),
    "model__min_samples_leaf": randint(1, 10)
}

# search = GridSearchCV(
# Trocar GridSearch Por Randomized para facilitar a busca com vários parametros.
search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=params,
    n_iter=10,
    scoring='neg_mean_absolute_error',
    cv=5,
    verbose=2,
    n_jobs=4,
    random_state=1
).fit(X=x, y=y)

results = pd.DataFrame(search.cv_results_)
results["mae"] = -results["mean_test_score"]
#{'model__max_depth': 10, 'model__max_features': np.float64(0.4027291235719791), 'model__min_samples_leaf': 1, 'model__min_samples_split': 3, 'model__n_estimators': 560}

print("Melhores parametros:")
print(search.best_params_)
print(f"Melhor MAE: {-search.best_score_}")



model = search.best_estimator_

test_data = pd.read_csv('E:/Programação/Kaggle/datasets/home-data-for-ml-course/test.csv')

ids = test_data['Id']
test_data.drop(columns=["Id"], inplace=True)

preds = model.predict(test_data)

submission = pd.DataFrame({
    "Id": ids,
    "SalePrice": preds
})
submission.to_csv('E:/Programação/Kaggle/datasets/home-data-for-ml-course/submission.csv', index=False)