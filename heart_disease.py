import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV , cross_val_score
from sklearn.metrics import class_likelihood_ratios ,confusion_matrix ,classification_report,accuracy_score
from sklearn.linear_model import LogisticRegression


df =pd.read_csv("nhanes_2017_2018_heart_disease_prediction.csv")

x =df.drop("heart_disease",axis=1)
y =df["heart_disease"]

x_train ,x_test ,y_train ,y_test =train_test_split(x,y,test_size=0.2,random_state=42)


model =RandomForestClassifier(n_estimators=200,max_depth=5,random_state=42,class_weight="balanced")
model.fit(x_train,y_train)
y_pred =model.predict(x_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))


param_grid = {"n_estimators" : [100,200,300],
              "max_depth":[3,5,10,None],
              "class_weight":["balanced"]}
grid =GridSearchCV(RandomForestClassifier(random_state=42),param_grid,cv=5,scoring="recall")
grid.fit(x_train,y_train)
y_pred =grid.predict(x_test)
print(grid.best_params_)
print(grid.best_score_)

best_model =grid.best_estimator_
y_pred =best_model.predict(x_test)

print(class_likelihood_ratios(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


model =LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
score =cross_val_score(model,x,y,cv=5)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(score.mean())
