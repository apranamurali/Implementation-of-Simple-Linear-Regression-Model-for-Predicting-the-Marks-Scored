# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program
```
 /*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: APARNA M
RegisterNumber:  212223220008
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:
df.head()

![image](https://github.com/user-attachments/assets/13ae1885-5b00-434e-a7e0-3bed24c56e59)

df.tail()

![image](https://github.com/user-attachments/assets/0fe5ca50-4f9b-46a2-936b-40c48e6a2e03)

Array value of X

![image](https://github.com/user-attachments/assets/9212801c-a111-4395-aa09-467321cd6dfd)

value of Y prediction


![image](https://github.com/user-attachments/assets/2646df99-c389-49f3-adc6-9c5ce50dd5fe)


Array values of Y test

![image](https://github.com/user-attachments/assets/62c8fdaa-0838-45c9-9958-730e996c1672)

Training Set Graph


![image](https://github.com/user-attachments/assets/0329fa43-8e16-4de0-b1f7-ef96ca5705da)

Test Set Graph


![image](https://github.com/user-attachments/assets/41b70d21-6037-46b0-9d22-86debc20436f)


Values of MSE,MAEAnd RMSE

![image](https://github.com/user-attachments/assets/4719f1a4-0b38-457a-a59c-9dc8fc897b67)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
