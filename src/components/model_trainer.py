import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import model_creation
import pandas as pd
import joblib

try:
    x_trans = pd.read_csv('artifacts/x_trans.csv')
    y_trans = pd.read_csv('artifacts/y_trans.csv')
    logging.info('data read from artifacts')

    model,_ = model_creation(x_trans,y_trans)

    joblib.dump(model,os.path.join('artifacts','mushroom_final_model.pkl'))
    logging.info('model saved in artifacts for usage.....')

except Exception as e:
    raise CustomException(e,sys)


