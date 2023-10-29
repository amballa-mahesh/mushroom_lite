import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression,Lasso
from sklearn.feature_selection import SelectFromModel
import joblib

def data_cleaning(data):
  #reading data 
  df = pd.read_csv(data)
  print('data read')

  # droping unecessary features
  df.drop(['veil-type','stalk-root'],axis =1, inplace=True)
  print('unwanted columns dropped')

  # converting dependent variable to numerics
  df['class'].replace({'p':0,'e':1},inplace = True)

  # data spliting to dependent and independent variables
  y = df['class']
  x = df.drop('class',axis =1)
  print('spliting data to independent and dependent variables')

  #one hot encoding of all features
  x_dummed = pd.get_dummies(x,drop_first=True)
  print('Onehot encoding done')

  # dimentionality reduction by finding the highly corelated columns
  columns = list(x_dummed.columns)
  corr_columns = []
  for i in range(0,90):
    for j in range(i,90):
      corr_value = x_dummed[columns[i]].corr(x_dummed[columns[j+1]])      
      if corr_value>=0.80:
        corr_columns.append(columns[j+1])
  corr_columns = set(corr_columns) 
  print('highly corelated columns found')
  x_dummed.drop(columns= corr_columns,axis =1, inplace =True)

  # dimentionality reduction using lasso
  feature_selection = SelectFromModel(Lasso(alpha = 0.001))
  feature_selection.fit(x_dummed,y)
  features = x_dummed.columns[feature_selection.get_support()]
  print('no. of features selected: ',len(features),'\n')
  print(features,'\n')
  
  x_new = x_dummed[features]
  print('highly corelated columns dropped')
  print('final data prepared for training')

  print(x_new.shape)
  return x_new,y

x_final,y_final= data_cleaning('D:/Git Data/mushroom_lite/files_main/mushrooms.csv')
print(x_final.shape)
print(y_final.shape)


def model_creation(x,y):

  # data spliting to dependent and independent variables
  x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,
                                                   random_state =123,stratify =y)

  #logistic regression
  estimator = LogisticRegression(solver = 'newton-cg')
  C = np.array([0.1,1,10,100])
  param_grid = {'C':C}
  neigh = GridSearchCV(estimator,param_grid,cv = 5,scoring= "roc_auc",
                       return_train_score=True,verbose =2)
  neigh.fit(x_train,y_train)
  model = LogisticRegression(C =neigh.best_params_['C'], solver = 'newton-cg')
  model.fit(x_train,y_train)
  
  return model

x_train,x_test,y_train,y_test = train_test_split(x_final,y_final,test_size = 0.2, random_state =123,stratify =y_final)

model = model_creation(x_final,y_final)

y_pred = model.predict(x_test)

print(confusion_matrix(y_test,y_pred))

joblib.dump(model,'D:/Git Data/mushroom_lite/artifacts/mushroom_final_model.pkl')

