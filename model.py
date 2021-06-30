##Load and preprocess data

import pandas as pd
inputFolder = '/home/deniz/Documents/aws/MLtest/Iris2/'
df = pd.read_csv(inputFolder + 'iris.csv')
print(df.head())

##split data into features and target

X = df.loc[:, df.columns != 'variety']
print(X.head())
y = df['variety']
print(y.head())

##Split data into train and test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25)
print(X_train.shape)
print(X_train.head())
print(X_test.shape)
print(X_test.head())
print(y_train.shape)
print(y_train.head())
print(y_test.shape)
print(y_test.head())

##Model creation

from sklearn.ensemble import RandomForestClassifier
#create object of RandomForestClassifier 
model = RandomForestClassifier()

##Training

#train model
model.fit(X_train, y_train)
#print score
model.score(X_train,y_train)
#Prediction
#predict X_test data
predictions = model.predict(X_test)
predictions[:10]

##Scoring
#Print accuracy, confustion_matrix and classification_report

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

##Saving model
#We are using joblib model to serialize Python object. joblib.dump() will allow to save Python object on disk.
import joblib

#save model in output directory
joblib.dump(model,'/home/deniz/Documents/aws/MLtest/Iris2/randomforest_model.pkl')
#Predict with new data
import numpy as np
test_data = [5.1, 3.2, 1.5, 0.4]

#convert test_data into numpy array
test_data = np.array(test_data)
#reshape
test_data = test_data.reshape(1,-1)
print(test_data)

##Load trained model
#declare path where you saved your model
outFileFolder = '/home/deniz/Documents/aws/MLtest/Iris2/'
filePath = outFileFolder + 'randomforest_model.pkl'

#open file
file = open(filePath, "rb")

#load the trained model
trained_model = joblib.load(file)
#Predict with trained model
prediction = trained_model.predict(test_data)
print(prediction)
