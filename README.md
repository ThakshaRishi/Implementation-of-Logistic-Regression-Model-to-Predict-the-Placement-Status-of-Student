# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required packages and print the present data
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. 
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Thaksha Rishi
RegisterNumber:  212223100058
*/

import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
```


<img width="1140" height="194" alt="image" src="https://github.com/user-attachments/assets/d9a8ae40-96ec-4fa9-9cfa-2d6e6c9e058b" />

```

data1=data.copy()
data1=data1.drop(["sl_no","salary"], axis=1)
data1.head()
```


<img width="1016" height="188" alt="image" src="https://github.com/user-attachments/assets/fd2b711a-ca95-4039-bf78-953f64f69555" />


```
data1.isnull().sum()
```


<img width="220" height="298" alt="image" src="https://github.com/user-attachments/assets/cee3e216-15f9-484b-95f7-3ea735a659b5" />

```
data1.duplicated().sum()
```


<img width="249" height="32" alt="image" src="https://github.com/user-attachments/assets/90f0d5b9-1c68-4fbd-8587-58af44f31831" />

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1 ["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
```


<img width="930" height="425" alt="image" src="https://github.com/user-attachments/assets/4b1caafe-e2b5-4409-82be-3ddcf839279c" />

```
x=data1.iloc[:,:-1]
x
```


<img width="861" height="433" alt="image" src="https://github.com/user-attachments/assets/6b4488d8-279a-45f2-af98-9f4e9ea03e80" />

```
y=data1["status"]
y
```


<img width="398" height="259" alt="image" src="https://github.com/user-attachments/assets/6d7ef140-6ca4-43f1-8512-bbf2ef215647" />

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression (solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
```


<img width="713" height="53" alt="image" src="https://github.com/user-attachments/assets/92f1e669-de24-4273-9420-02bd27866077" />

```
from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test,y_pred) 
accuracy
```


<img width="218" height="32" alt="image" src="https://github.com/user-attachments/assets/46474fc0-a070-4e39-aa6a-e6cb899b4534" />

```
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
```


<img width="565" height="193" alt="image" src="https://github.com/user-attachments/assets/b12c5958-26f7-49bb-bef0-a8636f595e2e" />

```
x_new=pd.DataFrame([[1,80,1,90,1,1,90,1,0,85,1,85]],columns=['gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s','degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p'])
print('Name: Thaksha Rishi')
print('Reg No: 212223100058')
lr.predict(x_new)
```


<img width="287" height="77" alt="image" src="https://github.com/user-attachments/assets/82cc276f-b341-4e75-a329-ed63b5028f37" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
