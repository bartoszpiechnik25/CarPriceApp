from joblib import load
import numpy as np

ml_parameters = list(load('files\column_names'))

def read_data():
    parameters_dict = dict(zip(ml_parameters,
     np.zeros(shape=len(ml_parameters), dtype=np.int64)))
    print('------------------------------------------------------------------')
    attr = ['mark', 'model', 'year', 'mileage', 'engine', 'province', 'city']
    user_data = []
    user_data.append(input('Specify mark of the car (mandatory): '))
    user_data.append(input('Specify model of the car (mandatory): '))
    user_data.append(int(input('Specify production year'\
    'of the car (mandatory): ')))
    user_data.append(int(input('Specify car mileage in KM (mandatory): ')))
    user_data.append(input('Specify fuel type (mandatory): '))
    user_data.append(float(input('Specify engine size '\
    '(2.0 is equivalent to 1998):')))
    user_data.append(input('Specify Polish province (example Małopolskie,'\
     'Kujawsko-pomorskie): '))
    user_data.append(input('Specify Polish city (example Kraków): '))

    for i in range(len(user_data)):
        if str(user_data[i]) not in ml_parameters and str(user_data[i]) in ('model', 'mark', 'province'):
            print(f'There is not enough data for a given {attr[i]},'\
             'predictions may be skewed')
        else:
            if isinstance(user_data[i], (int,float)):
                parameters_dict[attr[i]] = user_data[i]
            else:
                parameters_dict[user_data[i]] = 1

    # print(parameters_dict)
    
read_data()