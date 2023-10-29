<<<<<<< HEAD

Readme file.

This the Mushroom classifier.

Steps involved in creating this model are - 

EDA - 

reading data using Pandas
droping unecessary features
converting dependent variable to numerics
data spliting to dependent and independent variables
one hot encoding of all features
dimentionality reduction by finding the highly corelated columns
dimentionality reduction using lasso
create the dataframe using the features extracted from lasso.


Model Creation-

data spliting to dependent and independent variables
Logistic Regression, Naive Bayes, SVC, KNN, Decision Tree, Randomforest,SGD Classifier, Ada Boost, XG boost, Gradient boost.
Use GridsearchCV to hyper tune the models.
By using the ensemble technique (voting classifier) we ensemble all the best resulted models.
Use the voting classifier as the final model, train and evaluate the model using classification report or accuracy score or confusion matrix.


Prediction:
using the voting classifier perdict the test data
finding the model performance using the accuracy score, confusion matix and classification report.

Creation of User GUI-

Using the flask library we created the use GUI with HTML and CSS.
Deploy this model in local server.
get the values of the feilds selected by the user by flask
create the data frame and process the same to model.
get the predictions from the model.
return that back to user.

Using the logging

We will write back the logs to the logs.log file

Updating the data to mysql.

from the front end user interface get the values of selected feilds and save them back to local database by python mysql connector.
downloadt the data from the database and share....


=======
Please this file
>>>>>>> 35a0fe274b8c4798093b9a92c5571d64e63502ff
