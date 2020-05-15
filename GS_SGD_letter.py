import numpy
import pandas
import random
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
# inserting data

df=pandas.read_csv( 'datasets/letter.csv')
array=df.values
#Separating data into input and output components

X=array[0:4000,1:17]
Y=array[0:4000,0]

scaler=MinMaxScaler(feature_range=(0,1))
rX=scaler.fit_transform(X)

#clf = SGDClassifier(shuffle=True, loss='log')

params = {
    "loss" : ["hinge", "log", "squared_hinge", "modified_huber"],
    "alpha" : [0.0001, 0.001, 0.01, 0.1],
    "penalty" : ["l2", "l1", "none"],
}

model = SGDClassifier(max_iter=1000)
clf = GridSearchCV(model, param_grid=params)

n_iter = 2
for i in range(n_iter):
    x_train, x_test, y_train, y_test = train_test_split(rX, Y, test_size=0.3)  # splitting data
    clf.fit(x_train, y_train)
    clf_score = clf.score(x_test, y_test)
    print(clf_score)
    print(clf.best_estimator_)
