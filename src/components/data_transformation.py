import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import data_cleaning
import pandas as pd


try:
    data = 'artifacts/raw.csv'
    logging.info('Data Loaded')
    x_trans,y_trans = data_cleaning(data)
    x_trans.to_csv(os.path.join('artifacts','x_trans.csv'), index =False, header =True)
    y_trans.to_csv(os.path.join('artifacts','y_trans.csv'), index =False, header =True)
    logging.info('data transformed and saved in artifacts folder')

except Exception as e:
    raise CustomException(e,sys) 