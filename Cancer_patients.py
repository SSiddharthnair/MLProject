# This dataset is about determining the level of each cancer patients
# It was also observed that the dataset is labelled hence we must use supervised learning algorithm and with the output variable being categorical, we can try classification algorithm.
# Will be implementing KNN,SVM and logistic regression algorithms and comparing three accuracy scores

# 1. Start by importing all necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp
from sklearn import metrics

# 2. Import dataset
data = pd.read_csv('cancer_patients.csv')
df=pd.DataFrame(data)
print(df.to_string())

# Conversion of output from low/medium/high to 0/1/2 for easier calculations
df['Level'] = df['Level'].map({'Low': 0, 'Medium': 1, 'High': 2})


# 3. Select features for x and y
# first three columns are not required
x =df.iloc[:, 3:-1].values # independent variable
y =df.iloc[:, -1].values # dependant variable


# 4. Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# 5. Since the values are in different decimal scales, we do feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# 6. Fitting Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs', max_iter=10000)
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)
df2=pd.DataFrame(x_test)
print(df2.to_string)


# 7. Predicting the Test set results
df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df2.to_string())


# 8. Evaluating Algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
acc1=metrics.accuracy_score(y_test, y_pred)*100
print("Accuracy:",acc1)


# 9. Fitting K-Nearest Neighbour Algorithm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors=4, metric='minkowski', p=2 )
classifier.fit(x_train, y_train)


# 10. Predicting the Test set results
y_pred= classifier.predict(x_test)
print("Prediction comparison")
ddf=pd.DataFrame({"Y_test":y_test,"Y-pred":y_pred})
print(ddf.to_string())


# 11. Evaluating Algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
acc2 = accuracy_score(y_test, y_pred)*100
print('Accuracy: ',acc2)


# 12. Fitting Support Vector Machine Algorithm
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(x_train, y_train)


# 13. Predicting the Test set results
y_pred= classifier.predict(x_test)
df2=pd.DataFrame({"Actual Y_Test":y_test,"Prediction Data":y_pred})
print("prediction status")
print(df2.to_string())


# 14. Evaluating Algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
acc3 = accuracy_score(y_test, y_pred)*100
print('Accuracy:',acc3)

x = np.array(["Logistic", "KNN", "SVM"])
y = np.array([acc1,acc2,acc3])
mtp.bar(x, y, color="#4CAF50", width=0.5)
mtp.title('Comparison of Accuracy Scores')
mtp.show()


# Conclusion : We have observed no error and 100% Accuracy score using all three algorithms.Thus we can conclude that this is a great ML model.
