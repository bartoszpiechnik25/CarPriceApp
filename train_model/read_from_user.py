from joblib import load
import numpy as np

ml_parameters = list(load('files\column_names'))
mark_model = load('files\markmodel')
provinces = load('files\provinces')
fuel = ['Diesel', 'CNG', 'Gasoline', 'LPG', 'Hybrid', 'Electric']

def print_mark_model(lst, one_line_n: int, mark: str):
    if mark == 'mark':
        print('------------------List of marks-----------------')
    elif mark == 'model':
        print('------List of models of the selected brand------')
    elif mark == 'provinces':
        print('----------------List of provinces----------------')

    for number, element in enumerate(lst):
        if (number + 1) % one_line_n == 0:
            print(element, end='\n')
        else:
            print(element,  end= ', ')
    print('\n')

def read_data():
    parameters_dict = dict(zip(ml_parameters,
     np.zeros(shape=len(ml_parameters), dtype=np.int64)))

    attr = ['mark', 'model', 'year', 'mileage', 'fuel_type','vol_engine',
     'province', 'city']
    user_data = []
    
    print_mark_model(list(mark_model.keys()), 6, mark='mark')
    user_data.append(input('Specify mark of the car (mandatory): '))

    print_mark_model(list(mark_model[user_data[0]]), 6, mark='model')
    user_data.append(input('Specify model of the car (mandatory): '))
    
    user_data.append(int(input('Specify production year'\
    'of the car (mandatory): ')))
    user_data.append(int(input('Specify car mileage in KM (mandatory): ')))

    print(' '.join(fuel))
    user_data.append(input('Specify fuel type (mandatory): '))
    user_data.append(float(input('Specify engine size '\
    '(2.0 is equivalent to 1998):')))

    print_mark_model(provinces, 4, mark='provinces')
    user_data.append(input('Specify Polish province (example Małopolskie,'\
     'Kujawsko-pomorskie): '))
    user_data.append(input('Specify Polish city (example Kraków): '))

    usr_data_dict = dict(zip(attr, user_data))
    for num, obj in enumerate(user_data):
        if isinstance(obj, (int, float)):
            parameters_dict[attr[num]] = obj
        else:
            parameters_dict[obj] = 1
    return np.array(parameters_dict.values()), usr_data_dict

    # print(parameters_dict)
if __name__ == '__main__':
    #print(mark_model)
    print(read_data())
