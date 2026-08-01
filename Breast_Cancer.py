import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix ,class_likelihood_ratios ,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV ,cross_val_score,  cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df =pd.read_csv("breast-cancer.csv")


df["diagnosis"] =df["diagnosis"].map({"M" :1,'B':0})
x =df.drop(["diagnosis","id"],axis=1)
y =df["diagnosis"]


x_train ,x_test ,y_train ,y_test =train_test_split(x,y,test_size=0.2,random_state=42)

model =RandomForestClassifier(random_state=42)
model.fit(x_train,y_train)
y_pred =model.predict(x_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(class_likelihood_ratios(y_test,y_pred))
print(classification_report(y_test,y_pred))
 
grid_p ={"n_estimators" : [100,200,300],
              "max_depth":[3,5,10,None],
              "class_weight":["balanced"]}
grid=GridSearchCV(RandomForestClassifier(random_state=42),grid_p,cv=5,scoring='recall')
grid.fit(x_train,y_train)
y_grid = grid.predict(x_test)

print(grid.best_estimator_)
print(grid.best_params_)
print(grid.best_score_)
y_best =grid.best_estimator_
y_pred =y_best.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

pipe =Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(max_iter=1000))])
pipe.fit(x_train,y_train)
y_pipe =pipe.predict(x_test)
print(accuracy_score(y_test,y_pipe))
print(confusion_matrix(y_test,y_pipe))

score= cross_val_score(model,x,y,cv=5)
scores =cross_validate(model,x,y,cv=5,scoring=["accuracy","precision","recall" ,"f1"])
print(scores)
print(score.mean())