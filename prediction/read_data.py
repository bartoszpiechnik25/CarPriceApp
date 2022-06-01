import numpy as np
from joblib import load

class ReadFromUser:

    def __init__(self, param_file, model_file, provinces_file) -> None:
        self.__columns = load(param_file)
        self.__models = load(model_file)
        self.__provinces = load(provinces_file)
        self.__fuel = ['Diesel', 'CNG', 'Gasoline', 'LPG', 'Hybrid', 'Electric']
        self.__user_data = []


    def __data_print(self, lst, one_line_n: int, name: str):
        """
        This function print data in specified order from list.
        """
        if name == 'mark':
            print('------------------List of marks-----------------')
        elif name == 'model':
            print('------List of models of the selected mark------')
        elif name == 'provinces':
            print('----------------List of provinces----------------')

        for number, element in enumerate(lst):
            if (number + 1) % one_line_n == 0:
                print(element, end='\n')
            else:
                print(element,  end= ', ')
        print('\n')


    def __read_mark(self):
        #marks is list containing every mark in dataset
        marks = list(self.__models.keys())

        #Print all available marks in data set
        self.__data_print(marks, 6, name='mark')

        #Read data from user
        inpt = input('Specify mark of the car: ')

        #While mark is not correct read from user
        while inpt not in marks:
            print("\n----Specify correct mark!!-----\n")
            self.__data_print(marks, 6, name='mark')
            inpt = input('Specify mark of the car listed before: ')
        
        #Add mark to user data
        self.__user_data.append(inpt)
    
    def __read_model(self):
        model = list(self.__models[self.__user_data[0]])
        self.__data_print(model, 6, name='model')

        inpt = input('Specify model of the car: ')

        while inpt not in model:
            print(f"\n----Specify correct {self.__user_data[0]} model!!-----\n")
            self.__data_print(model, 6, name='model')
            inpt = input('Specify mark of the car listed before: ')

        self.__user_data.append(inpt)

    def __read_year(self):
        inpt = input('Specify your car year of production: ')

        while not isinstance(inpt, int):
            try:
                inpt = int(inpt)
            except ValueError:
                print('Invalid value!\n')
                inpt = input('Specify correct number: ')
        
        self.__user_data.append(inpt)

    def __read_mileage(self):
        inpt = input('Specify car mileage in kilometers: ')

        while not isinstance(inpt, int):
            try:
                inpt = int(inpt)
            except ValueError:
                print('Invalid value!\n')
                inpt = input('Specify correct number: ')
        self.__user_data.append(inpt)

    def __read_engine(self):
        inpt = input('Specify car engine (20=.0 is 1998): ')

        while not isinstance(inpt, int):
            try:
                inpt = int(inpt)
            except ValueError:
                print('Invalid value!\n')
                inpt = input('Specify correct number: ')
        self.__user_data.append(inpt)

    def __read_fuel(self):
        fl = ', '.join(self.__fuel)
        print(fl)

        inpt = input('Specify fuel type: ')

        while inpt not in self.__fuel:
            print('Specify correct type!')
            print(fl)
            inpt = inpt = input('Specify valid fuel type: ')
        
        self.__user_data.append(inpt)

    def __read_province_city(self, city=False):
        
        if city:
            inpt = input('Specify city: ')

        else:
            self.__data_print(self.__provinces, 4, 'provinces')
            inpt = input('Specify province: ')
            while inpt not in self.__provinces:
                print("Invalid province!\n")
                self.__data_print(self.__provinces, 4, 'provinces')
                inpt = input('Specify correct province: ')
            
        self.__user_data.append(inpt)

    def read_all(self):
        print(len(self.__columns))
        self.__read_mark()
        self.__read_model()
        self.__read_year()
        self.__read_mileage()
        self.__read_fuel()
        self.__read_engine()
        self.__read_province_city(city=False)
        self.__read_province_city(city=True)
    
    def car_data(self):
        attributes = ['mark', 'model', 'year', 'mileage', 'fuel_type',
        'vol_engine','province', 'city']

        print(f'Before dummy {len(self.__columns)}')

        dummy = dict(zip(self.__columns, np.zeros(shape=(len(self.__columns)),
         dtype=np.int32)))

        # dummy = {str(self.__columns[i]): 0 for i in range(len(self.__columns))}

        data = dict(zip(attributes[:-1], self.__user_data[:-1]))

        for i in range(len(attributes) - 1):
            if isinstance(self.__user_data[i], int):
                dummy[attributes[i]] = self.__user_data[i]
            else:
                dummy[self.__user_data[i]] = 1
        
        if self.__user_data[-1] in dummy.keys():
            dummy[self.__user_data[-1]] = 1
        
        return np.fromiter(dummy.values(), dtype=np.int32)



if __name__ == '__main__':

    read = ReadFromUser('files\column_names', 'files\markmodel',
     'files\provinces')
    read.read_all()
    val = read.car_data()
    print(f'Returned {len(val)}')
    print(val)