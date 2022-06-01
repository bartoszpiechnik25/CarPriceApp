from read_data import ReadFromUser
import lightgbm as lgb

model = lgb.Booster(model_file='files\lgbm_model.txt')

user = ReadFromUser('files\column_names', 'files\markmodel', 'files\provinces')
user.read_all()
values = user.car_data()

car_price = model.predict([values])
print(car_price)