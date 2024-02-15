# This dataset contains transactions made by credit cards in September 2013 by European cardholders.
# It was observed that the dataset is labelled and hence we must use supervised learning algorithm.
# I am implementing Naïve Bayes Classifier Algorithm here


# 1. Start by importing all necessary modules
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import metrics


# 2. Import dataset
data = pd.read_csv('D:/Data Analytics/ML/Project/creditcard.csv')
df=pd.DataFrame(data)
print(df.to_string())


# 3. Select features for x and y
x =df.iloc[:, 1:-1].values # independent variable
y =df.iloc[:, -1].values # dependant variable


# 4. Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# 5. Feature Scaling
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)


# 6. Fitting Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)


# 7. Predicting the Test set results
y_pred = classifier.predict(x_test)
print("------------PREDICTION----------")
df2=pd.DataFrame({"Actual Result-Y":y_test,"Prediction Result":y_pred})
print(df2.to_string())


# 8. Evaluating the Algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# 9. Evaluating Predictions
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))

# Conclusion : The Accuracy of this dataset is 97.84% using Naive Bayes Theorem, Hence this is considered as a great model.

