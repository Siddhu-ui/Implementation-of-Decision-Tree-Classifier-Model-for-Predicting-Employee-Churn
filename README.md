# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:

```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SIDDHARTH S 
RegisterNumber: 212224040317
*/
```
```
import pandas as pd
df=pd.read_csv("/content/Employee.csv")
print("data.head():")
df.head()
```
```
print("data.info()")
df.info()
```
```
print("data.isnull().sum()")
df.isnull().sum()
```
```
print("data value counts")
df["left"].value_counts()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print("data.head() for Salary:")
df["salary"]=le.fit_transform(df["salary"])
df.head()
```
```
print("x.head():")
x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
```

y=df["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
```
```

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
```
print("Data prediction")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
```

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plot_tree(dt,filled=True,feature_names=x.columns,class_names=['salary' , 'left'])
plt.show()
```

## Output:

![image](https://github.com/user-attachments/assets/40b3249c-d979-4c83-96e5-3aad54ea5e0c)
![image](https://github.com/user-attachments/assets/57df6bfc-9724-49f9-98fa-658d667327aa)  ![image](https://github.com/user-attachments/assets/f65e8667-317b-4a5d-8acc-47f9930dafe2)
![image](https://github.com/user-attachments/assets/41d380a8-6f9d-4e79-a696-dced9300cee5) ![image](https://github.com/user-attachments/assets/2a58e124-3d73-40cc-a6f8-6dd7904bac44)
![image](https://github.com/user-attachments/assets/37888b04-cc57-451c-aacc-1583b5543ed9)  ![image](https://github.com/user-attachments/assets/4171f5bb-a29d-4286-a1ac-30cd5733f0df)
![image](https://github.com/user-attachments/assets/c2af86c2-396e-410a-b49c-7be8ab617457)


![image](https://github.com/user-attachments/assets/d6f5246b-f7bf-4688-92dd-6ec05747401a)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
