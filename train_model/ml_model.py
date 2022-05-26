import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from joblib import dump

if __name__ == '__main__':
    #Load and prepare data for machine learning model.
    cars = pd.read_csv('train_model\Car_Prices_Poland_Kaggle.csv')
    cars.drop(columns=['Unnamed: 0', 'generation_name'], axis=1, inplace=True)
    cars_dumm = pd.get_dummies(data=cars,
     columns=['mark', 'model','fuel','city','province'])
    renaming ={cars_dumm.columns[i]: cars_dumm.columns[i].split('_')[1]\
         for i in range(4, len(cars_dumm.columns))}

    cars_dumm.rename(renaming, axis=1, inplace=True)
    print(cars_dumm.head())
    #Scaling data for better performance.
    sc = StandardScaler()
    scaled = sc.fit_transform(cars_dumm.values)
    y = scaled[:, 3]
    X = np.delete(scaled,3,axis=1)
    print('----X shape----',X.shape)
    print('----y shape----',y.shape)

    #Precomputed parameters with optuna module for LGBM algorithm
    parameters = {'n_estimators': 209,
     'reg_alpha': 4.579761912253526e-08,
     'reg_lambda': 0.05556953835133028,
     'num_leaves': 101,
     'colsample_bytree': 0.6736001169481415,
     'subsample': 0.7741897653550532,
     'min_child_samples': 19}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    #Create model with best params tested with optuna module and save it to file
    #for further use in app
    lgbm = LGBMRegressor(**parameters).fit(X_train, y_train)
    predicition = lgbm.predict(X_test)
    print(mean_squared_error(y_test, predicition))
    print(r2_score(y_test, predicition))
    dump(lgbm, 'files\cars_ml_model')
    dump(cars_dumm.drop('price', axis=1).columns, 'files\column_names')
