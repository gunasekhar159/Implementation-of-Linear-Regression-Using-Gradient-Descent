# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the linear regression using gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Use the standard libraries in python for Gradient Design.
2.Upload the dataset and check any null value using .isnull() function. 3.Declare the default values for linear regression.
3.Calculate the loss usinng Mean Square Error.
4.Predict the value of y.
5.Plot the graph respect to hours and scores using scatter plot function. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: M.GUNASEKHAR
RegisterNumber:212221240014  
*/

import pandas as pd
import matplotlib.pyplot as plt 
dataset=pd.read_csv('/content/student_scores - student_scores (1).csv')
dataset.head() 
X=dataset.iloc[:,:-1].values 
y=dataset.iloc[:,:1].values
print(X)
print(y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours v/s scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
y_pred=regressor.predict(X_test)
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours v/s scores(test set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
```

## Output:
![linear regression using gradient descent](sam.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
