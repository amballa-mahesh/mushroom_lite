from flask import Flask, redirect, url_for,render_template,request
import numpy as np
import joblib 
from joblib import dump, load
import pandas as pd
import logging
import mysql.connector
import cassandra
from datetime import datetime

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

# cloud_config= {
#   'secure_connect_bundle':'secure-connect-mushroomdb.zip'
# }
# auth_provider = PlainTextAuthProvider('KZmbqRbiNdUgPWYrqLSGxQFm',
#                                       '1Y6.GFpvbK-p0T_obq79hi.kz,wPW3Pxk7C4ZfSkysXMvcgrbs79HYde5kLQNijYhHrRkp.2U0LXNekO4_2T3uD,A1GMuASwXeWZF2_GZ5Hwd-2Kf5AyM6P-C7t-+yGG')

# cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
# session = cluster.connect('results')

# session.execute("insert into predictions(cap_color,odor,gill_spacing,gill_size,gill_color,stalk_shape,stalk_surface_above_ring,stalk_surface_below_ring,stalk_color_above_ring,stalk_color_below_ring,ring_type,spore_print_color,population,habitat,result) values('b','b','b','b','b','b','b','b','b','b','b','b','b','b','b')")
# session.execute("insert into predictions(cap_color,odor,gill_spacing,gill_size,gill_color,stalk_shape,stalk_surface_above_ring,stalk_surface_below_ring,stalk_color_above_ring,stalk_color_below_ring,ring_type,spore_print_color,population,habitat,result,time) values('c','d','a','a','a','a','a','a','a','a','a','a','a','a','k',toTimeStamp(now()))")
# sql confuguration

mydb = mysql.connector.connect(host = "localhost", user = "root", passwd = "1234",auth_plugin='mysql_native_password', database = 'mushroom_pred_db')
my_cursor = mydb.cursor()


logging.basicConfig(filename= 'files_main\logs.log',
                    filemode = 'a',
                    format = '%(asctime)s %(levelname)s-%(message)s',
                    datefmt= '%Y-%m-%d %H:%M:%S',
                    level = logging.DEBUG)

logging.info('libraries loaded...')

     
loaded_model = joblib.load('files_main\mushroom_final_model.pkl')
logging.info('Model Loaded..')

app = Flask(__name__)



# @app.route('/user')
# def welcome():
#     return render_template('welcome.html')


@app.route('/')
def welcome_user():
    return render_template('index.html')



@app.route('/submit', methods = ['POST','GET'])
def submit(): 
     features = ['cap-color_n', 'odor_c', 'odor_f', 'odor_l', 'odor_m', 'odor_n',
       'odor_p', 'odor_s', 'odor_y', 'gill-spacing_w', 'gill-size_n',
       'gill-color_n', 'gill-color_w', 'stalk-shape_t',
       'stalk-surface-above-ring_k', 'stalk-surface-above-ring_s',
       'stalk-surface-below-ring_s', 'stalk-surface-below-ring_y',
       'stalk-color-above-ring_o', 'stalk-color-below-ring_w',
       'stalk-color-below-ring_y', 'ring-number_t', 'ring-type_f',
       'ring-type_l', 'ring-type_p', 'spore-print-color_k',
       'spore-print-color_n', 'spore-print-color_r', 'population_n',
       'population_v', 'habitat_p', 'habitat_u', 'habitat_w']
       
     data = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
     logging.info('Empty numpy array created')

     if request.method == 'POST':
          cap_color  = request.form['cap-color']
          if cap_color == 'n':
               data[0]=1     

          odor       = request.form['odor']
          if odor == 'c':
               data[1]=1 
          elif odor == 'f':
               data[2]=1 
          elif odor == 'l':
               data[3]=1          
          elif odor == 'm':
               data[4]=1 
          elif odor == 'n':
               data[5]=1 
          elif odor == 'p':
               data[6]=1 
          elif odor == 's':
               data[7]=1 
          elif odor == 'y':
               data[8]=1 

          gill_spacing = request.form['gill-spacing']
          if gill_spacing == 'w':
               data[9]=1
          
          gill_size    =  request.form['gill-size']
          if gill_size == 'n':
               data[10] = 1
          
          gill_color   = request.form['gill-color']
          if gill_color == 'n':
               data[11] = 1
          elif gill_color == 'w':
               data[12] = 1

          stalk_shape   = request.form['stalk-shape']
          if stalk_shape == 't':
               data[13] = 1
          
          stalk_surface_above_ring = request.form['stalk-surface-above-ring']
          if stalk_surface_above_ring == 'k':
               data[14] = 1
          elif stalk_surface_above_ring == 's':
               data[15] = 1

          stalk_surface_below_ring = request.form['stalk-surface-below-ring']
          if stalk_surface_below_ring == 's':
               data[16] = 1
          elif stalk_surface_below_ring == 'y':
               data[17] = 1

          stalk_color_above_ring = request.form['stalk-color-above-ring']
          if stalk_color_above_ring == 'o':
               data[18] = 1

          stalk_color_below_ring = request.form['stalk-color-below-ring']
          if stalk_color_below_ring == 'w':
               data[19] = 1
          elif stalk_color_below_ring == 'y':
               data[20] = 1
     
          ring_number = request.form['ring-number']
          if ring_number == 't':
               data[21] = 1
          
          ring_type = request.form['ring-type']
          if ring_type == 'f':
               data[22] = 1
          elif ring_type == 'l':
               data[23] = 1
          elif ring_type == 'p':
               data[24] = 1
          
          spore_print_color = request.form['spore-print-color']
          if spore_print_color == 'k':
               data[25] = 1
          elif spore_print_color == 'n':
               data[26] = 1
          elif spore_print_color == 'r':
               data[27] = 1

          population = request.form['population']
          if population == 'n':
               data[28] = 1
          elif population == 'v':
               data[29] = 1

          habitat = request.form['habitat']
          if habitat == 'p':
               data[30] = 1
          elif habitat == 'u':
               data[31] = 1
          elif habitat == 'w':
               data[32] = 1

     print(len(data))
     logging.info('Date Captured from web api..')
     test = pd.DataFrame([data],columns =features)
     logging.info('Data frame created..')
     print(test.shape)   

     logging.info('Data sent to model for prediction..')
     y_pred = loaded_model.predict(test)
     res = ""
     if y_pred == 0:
          res =  'Posionous Mushroom!'
     else:
          res = 'Edible Mushroom!'
     print(res)
     logging.info('Prediction done sucessfully')
     logging.info('Predcited as - '+res)

     # for local sql data base data fleching 
     if cap_color == "":
          cap_color= 'oth'
     if odor == '':
          odor = 'oth'
     if gill_spacing == "":
          gill_spacing= 'oth'
     if gill_size == "":
          gill_size = 'oth'
     if gill_color == "":
          gill_color = 'oth'
     if stalk_shape == "":
          stalk_shape = 'oth'
     if stalk_surface_above_ring == "":
          stalk_surface_above_ring = 'oth'
     if stalk_surface_below_ring=="":
          stalk_surface_below_ring = 'oth'
     if stalk_color_above_ring == "":
          stalk_color_above_ring = 'oth'
     if stalk_color_below_ring=="":
          stalk_color_below_ring = ''
     if ring_type == "":
          ring_type = 'oth'
     if spore_print_color =="":
          spore_print_color= 'oth'
     if population == "":
          population = 'oth'
     if habitat  == "":
          habitat = 'oth'

     my_cursor.execute("insert into predictions(cap_color,odor,gill_spacing,gill_size,gill_color,stalk_shape,stalk_surface_above_ring,stalk_surface_below_ring,stalk_color_above_ring,stalk_color_below_ring,ring_type,spore_print_color,population,habitat,result) values('{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}');"
                       .format(cap_color,odor,gill_spacing,gill_size,gill_color,stalk_shape,stalk_surface_above_ring,stalk_surface_below_ring,stalk_color_above_ring,stalk_color_below_ring,ring_type,spore_print_color,population,habitat,res))
     mydb.commit()  

     time = datetime.now().ctime()
     

     
     # session.execute("insert into predictions(cap_color,odor,gill_spacing,gill_size,gill_color,stalk_shape,stalk_surface_above_ring,stalk_surface_below_ring,stalk_color_above_ring,stalk_color_below_ring,ring_type,spore_print_color,population,habitat,result,time) values('{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}');"
     #                   .format(cap_color,odor,gill_spacing,gill_size,gill_color,stalk_shape,stalk_surface_above_ring,stalk_surface_below_ring,stalk_color_above_ring,stalk_color_below_ring,ring_type,spore_print_color,population,habitat,res,time))

     #redering template...   
     return render_template('result.html',result = y_pred)
     


if __name__== '__main__':
    app.run(debug = True)



 