#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# importing important packages
import pandas as pd
#import numpy as np
from scipy.stats import mode

# reading the dataSet
df = pd.read_csv("data/trainFeatures.csv")
df1 = pd.read_csv("data/trainLabels.csv", header=None, names=['Labels'])
test_data = pd.read_csv("data/testFeatures.csv")

# combine trainFeatures and trainLabels
train_data = df.join(df1)
    
# count the error strings in each  column
def num_missing(x):
    cnt = 0
    for d in x:
        if str(d).strip() == '?':
            cnt = cnt + 1
    return cnt

# print the error strings in each column
print("Missing values per column:")  
print(train_data.apply(num_missing, axis=0))  

#transfer the data format from string to int
from sklearn import preprocessing
cat_cols = ['workclass','Marital-status','education','occupation','relationship','race','sex','native-country']
for col in cat_cols:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_data[col].values.astype('str')))
    train_data[col] = lbl.transform(list(train_data[col].values.astype('str')))
    
#fill the missing value by mode
occupation = mode(train_data['occupation']).mode[0] #get mode value
for i in range(len(train_data['occupation'])):      #change the null value to mode
    if str(train_data['occupation'][i]).strip() == '0':
        train_data['occupation'][i] = occupation

workclass = mode(train_data['workclass']).mode[0]
for i in range(len(train_data['workclass'])):
    if str(train_data['workclass'][i]).strip() == '0':
        train_data['workclass'][i] = workclass

native_country = mode(train_data['native-country']).mode[0]
for i in range(len(train_data['native-country'])):
    if str(train_data['native-country'][i]).strip() == '0':
        train_data['native-country'][i] = native_country

#reprint the missing value
print(train_data.apply(num_missing, axis=0))  # axis=0 代表函数应用于每一列

#delete the noise      
train_data = train_data.drop(17026)

# delete the duplicate row
train_data.drop_duplicates(keep='first')
    
# split dataSet into train and test
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split 
train, test = train_test_split(train_data, test_size=0.2, random_state=0)

# split train and test into labels and dataset
x_train = train.drop('Labels', axis=1)  
y_train = train['Labels']

x_test = test.drop('Labels', axis=1)
y_test = test['Labels']

# train the adaboost mode
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(random_state=1)
model.fit(x_train, y_train)
print("The accuracy of Adaboost model is: ")
print(model.score(x_test,y_test))


# part of use testing data to get the prediction 

# print error string of each column
print("Missing values per column:") 
print(test_data.apply(num_missing, axis=0))  

#transfer the string value to int
for col in cat_cols:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(test_data[col].values.astype('str')))
    test_data[col] = lbl.transform(list(test_data[col].values.astype('str'))) 
    
#use mode to fill the missing data in testing data 
occupation = mode(test_data['occupation']).mode[0]
for i in range(len(test_data['occupation'])):
    if str(test_data['occupation'][i]).strip() == '0':
        test_data['occupation'][i] = occupation

workclass = mode(test_data['workclass']).mode[0]
for i in range(len(test_data['workclass'])):
    if str(test_data['workclass'][i]).strip() == '0':
        test_data['workclass'][i] = workclass

native_country = mode(test_data['native-country']).mode[0]
for i in range(len(test_data['native-country'])):
    if str(test_data['native-country'][i]).strip() == '0':
        test_data['native-country'][i] = native_country

# use the train mode to predict the label of testing data     
pred=model.predict(test_data)
#np.savetxt('A2_qhuangak_20548333_prediction.csv', pred, delimiter = ',')

