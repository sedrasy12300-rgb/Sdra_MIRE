import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("train.csv")
df["Age"] =df["Age"].fillna(df["Age"].mean())
df["Embarked"] =df["Embarked"].fillna(df["Embarked"].mode()[0])
df =pd.get_dummies(df,columns=["Embarked"])
df =df.drop("Cabin",axis=1)
df["Sex"] =df["Sex"].map({'male' :  0,"female":1})


df["Title"] =df["Name"].str.extract("([A-Za-z]+)\.")
df =pd.get_dummies(df,columns=["Title"])
df["FamilySize"] = df["SibSp"] +df["Parch"] +1



df =df.drop(["PassengerId","Ticket","Name"],axis=1)

x =df.drop("Survived",axis=1)
y= df["Survived"]

x_train ,x_test ,y_train ,y_test =train_test_split(x,y,test_size=0.2,random_state=42)

model  =RandomForestClassifier(n_estimators=300,random_state=42)
model.fit(x_train ,y_train)
y_pred =model.predict(x_test)


accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
print(confusion_matrix(y_test,y_pred))
model2 =LogisticRegression(max_iter=1000)
model2.fit(x_train,y_train)
y_pred2  =model2.predict(x_test)
print(accuracy_score(y_test,y_pred2))
score =cross_val_score(model,x,y,cv =5)
print(score)
print(score.mean())



params ={"n_estimators" :[100,200,300],"max_depth": [3,5,10,None]}

grid =GridSearchCV(RandomForestClassifier(random_state=42),param_grid=params,cv =5)
grid.fit(x_train,y_train)
print(grid.best_params_)
print(grid.best_score_)

_best_model_ =grid.best_estimator_

_best_y_pred =_best_model_.predict(x_test)


print(accuracy_score(y_test,_best_y_pred))
print(confusion_matrix(y_test,_best_y_pred))



pipeline=Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(max_iter=1000))])
pipeline.fit(x_train,y_train)
y_pred =pipeline.predict(x_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
