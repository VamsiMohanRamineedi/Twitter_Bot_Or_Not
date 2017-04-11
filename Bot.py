# Categorizing verified, default_profile, default_profile_image features
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,5]=labelencoder_X.fit_transform(X[:,5])
X[:,7]=labelencoder_X.fit_transform(X[:,7])
X[:,8]=labelencoder_X.fit_transform(X[:,8])

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
classifier_DT = DecisionTreeClassifier(criterion = "entropy", random_state=0)
classifier_DT.fit(X_train, y_train)
prediction_DT = classifier_DT.predict(X_test)

accuracy_score(y_test, prediction_DT)

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_RF.fit(X_train, y_train)
prediction_RF = classifier_RF.predict(X_test)

accuracy_score(y_test, prediction_RF)
