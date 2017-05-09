#Import the libraries
import pandas as pd
import datetime
import numpy as np
import re
from sklearn.metrics import accuracy_score

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\parin\Desktop\training_data_2_csv_UTF.csv')
test_data = pd.read_csv(r'C:\Users\parin\Desktop\test_data_4_students.csv')
test_data = test_data.loc[:574,:]
    
# Cleaning the data. Getting rid of the quotations 
for i in range(0,len(dataset)):
    if dataset['created_at'][i].startswith('"'):
        dataset['created_at'][i] = dataset['created_at'][i][1:]
    if dataset['created_at'][i].endswith('"'):
        dataset['created_at'][i] = dataset['created_at'][i][0:-1]

                
# Selecting only required features from the training set 
dataset_new = dataset[["screen_name","description","name","followers_count","friends_count","listedcount","favourites_count","verified","statuses_count","created_at","bot"]]
dataset_new["description"] = dataset_new["description"].astype(str)
dataset_new["name"] = dataset_new["name"].astype(str)
for i in range(0, len(dataset_new)): 
    dataset_new["screen_name"][i] = dataset_new["screen_name"][i].lower()
    dataset_new["description"][i] = dataset_new["description"][i].lower()
    dataset_new["name"][i] = dataset_new["name"][i].lower()

#Training set
# If screen_name, name and description features have the word bot in it, it is assigned the value 1 else 0. 
X = dataset_new.iloc[:, :-1].values       
pattern=r"bot"
for i in range(0, len(dataset_new)):
    if re.search(pattern, X[i,0]):
        dataset_new["screen_name"][i]=1       
    else:
        dataset_new["screen_name"][i]=0
    X[i,0]= dataset_new["screen_name"][i]
    
    if re.search(pattern, X[i,1]):
        dataset_new["description"][i]=1       
    else:
        dataset_new["description"][i]=0
    X[i,1]= dataset_new["description"][i]

    if re.search(pattern, X[i,2]):
        dataset_new["name"][i]=1       
    else:
        dataset_new["name"][i]=0
    X[i,2]= dataset_new["name"][i]  
y = dataset_new.iloc[:,10].values # True results of the training set (bot field)

# Selecting only required features from the training set 
test_data_new = test_data[["screen_name","description","name","followers_count","friends_count","listed_count","favorites_count","verified","statuses_count","created_at","id"]]
test_data_new["description"] = test_data_new["description"].astype(str)
test_data_new["name"] = test_data_new["name"].astype(str)

# Dealing with missing values.
for i in range(0, len(test_data_new)):
    if test_data_new["followers_count"][i] == "None":
        test_data_new["followers_count"][i] = 0
    if test_data_new["friends_count"][i] == "None":
        test_data_new["friends_count"][i] = 0
    if test_data_new["listed_count"][i] == "None":
        test_data_new["listed_count"][i] = 0   
    if test_data_new["favorites_count"][i] == "None":
        test_data_new["favorites_count"][i] = 0 
    if test_data_new["statuses_count"][i] == "None":
        test_data_new["statuses_count"][i] = 0
    if test_data_new["verified"][i] == "None":
        test_data_new["verified"][i] = 'FALSE'
        
#Test set
# If screen_name, name and description features have the word bot in it, it is assigned the value 1 else 0.                    
for i in range(0, len(test_data_new)):
    test_data_new["screen_name"][i] = test_data_new["screen_name"][i].lower()
    test_data_new["description"][i] = test_data_new["description"][i].lower()
    test_data_new["name"][i] = test_data_new["name"][i].lower()
Z = test_data_new.iloc[:, :-1].values
                      
for i in range(0, len(test_data_new)):
    if re.search(pattern, Z[i,0]):
        test_data_new["screen_name"][i]=1       
    else:
        test_data_new["screen_name"][i]=0
    Z[i,0]= test_data_new["screen_name"][i]
      
    if re.search(pattern, Z[i,1]):
        test_data_new["description"][i]=1    
    else:
        test_data_new["description"][i]=0
    Z[i,1]= test_data_new["description"][i]
    
    if re.search(pattern, Z[i,2]):
        test_data_new["name"][i]=1  
    else:
        test_data_new["name"][i]=0
    Z[i,2]= test_data_new["name"][i]
    
# Finds ratio of statuses_count to the number of days since account creation for training and test set
dataset['created_at'] = pd.to_datetime(dataset['created_at'])
sc = dataset['statuses_count']
l = [None] * len(dataset['created_at'])
today = datetime.datetime.today()
for i in range(len(dataset['created_at'])):
    l[i] = (abs(dataset['created_at'][i] - today)).days
    l[i] = sc[i]/l[i]
    X[i,9] = l[i] 
        
test_data_new["created_at"]  = pd.to_datetime(test_data_new["created_at"])
sct = test_data_new['statuses_count']
m = [None] * len(test_data_new["created_at"])
for i in range(len(test_data_new["created_at"])):
    m[i] = (abs(test_data_new["created_at"][i] - today)).days
    m[i] = sct[i]/m[i]
    Z[i,9] = m[i]
    

# Encode categorical values for Verified field
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,7]=labelencoder_X.fit_transform(X[:,7])
Z[:,7]=labelencoder_X.fit_transform(Z[:,7])

#N = X[:,[0,1,2,3,4,5,6,7,9]]  #Different slices of training set to check accuracy
#M = Z[:,[0,1,2,3,4,5,6,7,9]]

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_RF.fit(X, y)
prediction_RF = classifier_RF.predict(Z)

test_data_new["bot"] = prediction_RF
test_data_new["id"] = test_data_new["id"].astype(np.int64) 
outputRF=test_data_new[["id","bot"]]
outputRF.to_csv('RFprediction.csv', index =False)

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
classifier_DT = DecisionTreeClassifier(criterion = "entropy", random_state=0)
classifier_DT.fit(X, y)
prediction_DT = classifier_DT.predict(Z)

test_data_new["bot"] = prediction_DT
test_data_new["id"] = test_data_new["id"].astype(np.int64) 
outputDT=test_data_new[["id","bot"]]
outputDT.to_csv('DTprediction.csv', index =False)

"""
# KFOLD Cross Validation Score
#Random Forest Classifier KFold
from sklearn.model_selection import KFold, cross_val_score
kf = KFold(n_splits=4, random_state=None, shuffle=False)
from sklearn.ensemble import RandomForestClassifier
#classifier_RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
print (cross_val_score(classifier_RF, X, y, cv=kf))

# Decision Tree classifier KFold
from sklearn.model_selection import KFold, cross_val_score
kf = KFold(n_splits=4, random_state=None, shuffle=False)
from sklearn.tree import DecisionTreeClassifier
#classifier_DT = DecisionTreeClassifier(criterion = "entropy", random_state=0)
print (cross_val_score(classifier_DT, X, y, cv=kf))

#Accuracy Score
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_RF.fit(X_train, y_train)
prediction_RF = classifier_RF.predict(X_test)
print("Random Forest")
print(accuracy_score(y_test, prediction_RF))

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
classifier_DT = DecisionTreeClassifier(criterion = "entropy", random_state=0)
classifier_DT.fit(X_train, y_train)
prediction_DT = classifier_DT.predict(X_test)
print("Decision Tree")
print(accuracy_score(y_test, prediction_DT))
"""
