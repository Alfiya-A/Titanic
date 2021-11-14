import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv("/Users/alfia/Desktop/Titanic/titanic.csv")

df.drop(['PassengerId','Name','Ticket','Fare','Cabin'],axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder
le_sex=LabelEncoder()
le_embarked=LabelEncoder()


df['Gender']=le_sex.fit_transform(df.Sex)
df['Embark']=le_embarked.fit_transform(df.Embarked)


df.drop(["Embarked","Sex"],axis=1,inplace=True)


df.Age.isnull().value_counts()

mean=df.Age.mean()
df.Age = df.Age.fillna(mean)



df.Age.isnull().value_counts()


df.Age=df.Age.apply(int)




X=df.drop("Survived",axis=1)
y=df.Survived


from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

x_train.shape, x_test.shape

from sklearn.linear_model import LogisticRegression

regressor=LogisticRegression()

regressor.fit(x_train,y_train)


pickle.dump(regressor,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

print(model.predict([[3,22,1,0,1,2]]))