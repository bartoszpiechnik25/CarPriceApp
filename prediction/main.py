from train_model.read_from_user import read_data
from joblib import load

regression_model = load('cars_ml_model')

var_predict, car_data = read_data()
print(var_predict)