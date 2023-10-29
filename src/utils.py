import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression,Lasso,SGDClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
import xgboost
from xgboost import XGBClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
import joblib
from src.logger import logging

def data_cleaning(data):
  #reading data 
  df = pd.read_csv(data)
  print('data read')
  logging.info('data read')

  # droping unecessary features
  df.drop(['veil-type','stalk-root'],axis =1, inplace=True)
  print('unwanted columns dropped')
  logging.info('unwanted columns dropped')


  # converting dependent variable to numerics
  df['class'].replace({'p':0,'e':1},inplace = True)

  # data spliting to dependent and independent variables
  y = df['class']
  x = df.drop('class',axis =1)
  print('spliting data to independent and dependent variables')
  logging.info('spliting data to independent and dependent variables')

  #one hot encoding of all features
  x_dummed = pd.get_dummies(x,drop_first=True)
  print('Onehot encoding done')
  logging.info('Onehot encoding done')

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
  logging.info('highly corelated columns found')

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
  logging.info('highly corelated columns dropped')
  logging.info('final data prepared for training')

  print(x_new.shape)
  return x_new,y

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
  model_lg = LogisticRegression(C =neigh.best_params_['C'], solver = 'newton-cg')
  model_lg.fit(x_train,y_train) 

  #Support vector machine
  estimator = SVC()
  C      = np.array([1,5])
  gamma  = np.array([0.01,0.1,1])
  kernel = ['poly','rbf']
  degree = np.array([1,2])
  param_grid = {'C':C,"gamma":gamma,'kernel':kernel,'degree':degree}
  neigh = GridSearchCV(estimator,param_grid,cv = 5,scoring ='roc_auc',
                       return_train_score=True,verbose = 2)  
  neigh.fit(x_train,y_train)
  model_svc =SVC(C= neigh.best_params_['C'], degree= neigh.best_params_['degree'],
                 gamma= neigh.best_params_['gamma'],
                 kernel= neigh.best_params_['kernel'])
  model_svc.fit(x_train,y_train)

  #Decision Tree
  estimator = DecisionTreeClassifier()
  min_samples_split = np.array([3,5,7])
  max_leaf_nodes   = np.array([10,15,20])
  param_grid = {'min_samples_split':min_samples_split,'max_leaf_nodes':max_leaf_nodes}
  neigh  = GridSearchCV(estimator,param_grid,cv =5, scoring ='roc_auc',
                        return_train_score =True,verbose =2) 
  neigh.fit(x_train,y_train)
  model_dc = DecisionTreeClassifier(max_leaf_nodes=neigh.best_params_['max_leaf_nodes'],
                                    min_samples_split=neigh.best_params_['min_samples_split'])
  model_dc.fit(x_train,y_train)

  #Random Forest
  estimator = RandomForestClassifier()
  min_samples_split = np.array([3,5])
  max_leaf_nodes    = np.array([15,20,25])
  n_estimators      = np.array([25,30])
  param_grid = {'min_samples_split':min_samples_split,'max_leaf_nodes':max_leaf_nodes,'n_estimators':n_estimators}
  neigh  = GridSearchCV(estimator,param_grid,cv =5, scoring ='roc_auc',
                        return_train_score =True,verbose =2)
  neigh.fit(x_train,y_train)
  model_rfc = RandomForestClassifier(max_leaf_nodes=neigh.best_params_['max_leaf_nodes']
                                     ,min_samples_split=neigh.best_params_['min_samples_split'],
                                     n_estimators=neigh.best_params_['n_estimators'])
  model_rfc.fit(x_train,y_train)

  # SGD Classifier
  estimator = SGDClassifier(penalty='l2',loss='log_loss',max_iter=500)
  alphas =np.array([0.00001,0.0001,0.001,0.01,0.1,1])
  param_grid = {'alpha':alphas}
  neigh  = GridSearchCV(estimator,param_grid,cv = 5,scoring = 'roc_auc',
                        return_train_score =True,verbose=2)
  neigh.fit(x_train,y_train)
  model_sgd = SGDClassifier(alpha=neigh.best_params_['alpha'], loss='log', max_iter=500)
  model_sgd.fit(x_train,y_train)

  # GradientBoostingClassifier
  estimator = GradientBoostingClassifier()
  learning_rate = np.array([0.001,0.01,0.1])
  n_estimators  = np.array([50,100])
  param_grid = {'learning_rate':learning_rate,'n_estimators':n_estimators}
  neigh = GridSearchCV(estimator,param_grid,cv=5,return_train_score =True,verbose=2)
  neigh.fit(x_train,y_train)
  model_gbc = GradientBoostingClassifier(learning_rate=neigh.best_params_['learning_rate'],
                                         n_estimators = neigh.best_params_['n_estimators'])
  model_gbc.fit(x_train,y_train)

  #XGBClassifier
  estimator = XGBClassifier()
  learning_rate = np.array([0.001,0.01,0.1])
  n_estimators  = np.array([30,40,50])
  param_grid = {'learning_rate':learning_rate,'n_estimators':n_estimators}
  neigh = GridSearchCV(estimator,param_grid,cv=5,
                       return_train_score =True,verbose=2)
  neigh.fit(x_train,y_train)
  model_xgb = XGBClassifier(learning_rate = neigh.best_params_['learning_rate'],
                            n_estimators = neigh.best_params_['n_estimators'])
  model_xgb.fit(x_train,y_train)

  # AdaBoostClassifier
  estimator = AdaBoostClassifier()
  learning_rate = np.array([0.1,0.2,0.3,0.4,0.5])
  n_estimators  = np.array([50,80,100])
  param_grid = {'learning_rate':learning_rate,'n_estimators':n_estimators}
  neigh = GridSearchCV(estimator,param_grid,cv=5,
                       return_train_score =True,verbose=2)
  neigh.fit(x_train,y_train)
  model_abc = AdaBoostClassifier(learning_rate = neigh.best_params_['learning_rate'],
                                 n_estimators = neigh.best_params_['n_estimators'])
  model_abc.fit(x_train,y_train)

  # BernoulliNB
  estimator=BernoulliNB()
  alpha = np.array([0.001,0.01,0.1])
  param_grid = {'alpha':alpha}
  neigh = GridSearchCV(estimator,param_grid,cv=5,
                       return_train_score =True,verbose=2)
  neigh.fit(x_train,y_train)
  model_bnb = BernoulliNB(alpha= neigh.best_params_['alpha'])
  model_bnb.fit(x_train,y_train)

  # KNeighborsClassifier
  estimator = KNeighborsClassifier(algorithm='auto', leaf_size=30)
  n_neighbors = np.array([5,15])
  param_grid = {'n_neighbors':n_neighbors}
  neigh = GridSearchCV(estimator,param_grid,cv=5,
                       return_train_score =True,verbose=2)
  neigh.fit(x_train,y_train)
  model_knn = KNeighborsClassifier(n_neighbors= neigh.best_params_['n_neighbors'])
  model_knn.fit(x_train,y_train)

  #checking scores of each model
  scores = {}
  score_lg = accuracy_score(y_test,model_lg.predict(x_test))
  scores['LG'] = round(score_lg,2)
  score_svc = accuracy_score(y_test,model_svc.predict(x_test))
  scores['SVC'] = round(score_svc,2)
  score_dc = accuracy_score(y_test,model_dc.predict(x_test))
  scores['DTC'] = round(score_dc,2)
  score_rfc = accuracy_score(y_test,model_rfc.predict(x_test))
  scores['RFC'] = round(score_rfc,2)
  score_sgd = accuracy_score(y_test,model_sgd.predict(x_test))
  scores['SGD'] = round(score_sgd,2)
  score_gbc = accuracy_score(y_test,model_gbc.predict(x_test))
  scores['GBC'] = round(score_gbc,2)
  score_xgb = accuracy_score(y_test,model_xgb.predict(x_test))
  scores['XGB'] = round(score_xgb,2)
  score_abc= accuracy_score(y_test,model_abc.predict(x_test))
  scores['ABC'] = round(score_abc,2)
  score_bnb = accuracy_score(y_test,model_bnb.predict(x_test))
  scores['BNB'] = round(score_bnb,2)
  score_knn = accuracy_score(y_test,model_knn.predict(x_test))
  scores['KNN'] = round(score_knn,2)
  print(scores)
  logging.info(scores)

  #visualization
  plt.figure(figsize= (14,4))
  plt.bar(scores.keys(),scores.values())
  plt.grid()
  plt.show()

  #Creation of Ensemble final Model
  estimators_all = [('RF', model_rfc),
                    ('LG', model_lg),
                    ('SGD', model_sgd),
                    ('SVC', model_svc),
                    ('GBC',model_gbc),
                    ('XGBC',model_xgb),
                    ('DTC',model_dc),
                    ('ABC',model_abc),
                    ('BNB',model_bnb),
                    ('KNN',model_knn)]
  model_final = VotingClassifier(estimators = estimators_all,
                                voting = 'hard')
  model_final.fit(x_train,y_train)
  return model_final,scores

