import numpy
import pandas
import random
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
# inserting data

df=pandas.read_csv( 'datasets/sensor.csv')
array=df.values
#Separating data into input and output components
s1=array[0:1000,0:48]
s2=array[7000:8000,0:48]
s2=array[12000:13000,0:48]
s3 =array[18000:19000,0:48]
s4=array[22000:23000,0:48]
s5=array[28000:29000,0:48]
s6=array[32000:33000,0:48]
s7=array[37000:38000,0:48]
s8=array[42000:43000,0:48]
s9=array[49000:50000,0:48]
s10=array[56000:57000,0:48]

X = numpy.concatenate((s1,s2,s3,s4,s5,s6,s7,s8,s9,s10), axis=0)
ss1=array[0:1000,48]
ss2=array[7000:8000,48]
ss2=array[12000:13000,48]
ss3=array[18000:19000,48]
ss4=array[22000:23000,48]
ss5=array[28000:29000,48]
ss6=array[32000:33000,48]
ss7=array[37000:38000,48]
ss8=array[42000:43000,48]
ss9=array[49000:50000,48]
ss10=array[56000:57000,48]
Y = numpy.concatenate((ss1, ss2, ss3, ss4, ss5, ss6, ss7, ss8, ss9, ss10), axis=0)

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
