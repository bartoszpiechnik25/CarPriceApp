from read_data import ReadFromUser
import lightgbm as lgb

model = lgb.Booster(model_file='CarPriceApp/files/lgbm_model.txt')

user = ReadFromUser('CarPriceApp/files/column_names',
                    'CarPriceApp/files/markmodel',
                    'CarPriceApp/files/provinces')
user.create_user_data()
user.show_results(model)