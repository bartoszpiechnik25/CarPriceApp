from read_data import ReadFromUser
import lightgbm as lgb

model = lgb.Booster(model_file='files\lgbm_model.txt')

user = ReadFromUser('files\column_names', 'files\markmodel', 'files\provinces')
user.create_user_data()
user.show_results(model)