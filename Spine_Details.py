# This data set is about identifying if a person is abnormal or normal using collected physical spine details/data.
# It is a Labelled dataset with Categorical output, Hence Decision Tree Classification Algorithm will be implemented


# 1. Start by importing all necessary modules
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import metrics


# 2. Import dataset
data = pd.read_csv('Dataset_spine.csv')
df=pd.DataFrame(data)
print(df.to_string())

# Conversion of output from good/bad to 1/0 for easier calculations
df['Class_att'] = df['Class_att'].map({'Abnormal': 1, 'Normal': 0})

# 3. Select features for x and y
x =df.iloc[:, :-1].values # independent variable
y =df.iloc[:, -1].values # dependant variable


# 4. Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)


# 5. Feature Scaling
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)


# 6. Fitting Decision Tree classifier to the training set
from sklearn.tree import DecisionTreeClassifier
classifier= DecisionTreeClassifier(criterion='gini', random_state=0)
classifier.fit(x_train, y_train)


# 7. Predicting the Test set results
y_pred= classifier.predict(x_test)
df2=pd.DataFrame({"Actual Y_Test":y_test,"Prediction Data":y_pred})
print("Prediction Result")
print(df2.to_string())


# 8. Evaluating Algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# 9. Evaluating Predictions
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))

# The Observed accuracy for this model is 80.77% Hence this is considered as a good model.
