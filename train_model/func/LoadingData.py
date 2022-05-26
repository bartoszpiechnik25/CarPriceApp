import pandas as pd
from IPython.display import display
from func.blueprints import plot_missing_data
import sys


class LoadData:
    """
    Reads your csv data from specified path and load it to
    pandas.DataFrame object.
    ---------------------------------------------------------
    If path does not exist, or you were given invalid name it
    will return -1.
    ---------------------------------------------------------"""
    def __init__(self, path):

        try:
            self.__path = path
            self.__df = pd.read_csv(path)
        except FileNotFoundError:
            print(f'Specified path -> {self.__path} does not exists')
            sys.exit(-1)

    def create_data(self):
        """
    This function will summarize your dataset and display:
    ------------------------------------------------------
    info: df.info()
    shape: df.shape
    head: df.head()
    describe: df.describe()
    number of NaN values: df.isna().sum()
    plot number of NaN on heatmap: sns.heatmap()
    number of duplicates: df.duplicated().sum()
    ------------------------------------------------------"""
        print(f'Shape of DataFrame is {self.__df.shape}\n')
        display(self.__df.info())
        print('\nHead of DataFrame:')
        display(self.__df.head())
        print('\nDataset statistics:')
        display(self.__df.describe())
        print('\nNumber of NaN values in DataFrame:')
        display(self.__df.isna().sum())
        print('\nNumber of duplicates in DataFrame:')
        display(self.__df.duplicated().sum())
        print('\nVisualisation of missing values in data set:')
        plot_missing_data(self.__df)
        return self.__df


if __name__ == '__main__':
    load = LoadData('titanic_train.csv')
    df = load.create_data()
    df.head()
    print(LoadData.create_data.__doc__)
    print(LoadData.__doc__)


