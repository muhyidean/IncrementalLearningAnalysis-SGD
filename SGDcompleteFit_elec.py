import numpy
import pandas
import random
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, recall_score, precision_score
# inserting data

df=pandas.read_csv( 'datasets/elec2.csv ')
array=df.values
#Separating data into input and output components
X=array[:,0:7]
Y=array[:,7]
scaler=MinMaxScaler(feature_range=(0,1))
rX=scaler.fit_transform(X)

clf = SGDClassifier(alpha=0.001, average=False, class_weight=None,
              early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
              l1_ratio=0.15, learning_rate='optimal', loss='modified_huber',
              max_iter=1000, n_iter_no_change=5, n_jobs=None, penalty='none',
              power_t=0.5, random_state=None, shuffle=True, tol=0.001,
              validation_fraction=0.1, verbose=0, warm_start=False)

results = numpy.zeros(shape=(30))
pre_results = numpy.zeros(shape=(30))
rec_results = numpy.zeros(shape=(30))
f1_results = numpy.zeros(shape=(30))

rmean = numpy.zeros(shape=(1))                      # get mean results for one chunk size
rstd = numpy.zeros(shape=(1))                       # get std results for one chunk size
pre_mean = numpy.zeros(shape=(1))                            # get mean results for one chunk size
rec_mean = numpy.zeros(shape=(1))                            # get mean results for one chunk size
f1_mean = numpy.zeros(shape=(1))                             # get mean results for one chunk size
comReport = numpy.zeros(shape=(5))
n_iter = 30
for i in range(n_iter):
    x_train, x_test, y_train, y_test = train_test_split(rX, Y, test_size=0.3)  # splitting data  70 / 30
    clf.fit(x_train, y_train)
    clf_score = clf.score(x_test, y_test)

    results[i] = clf_score
    weighted_prediction = clf.predict(x_test)
    f1_results[i] = f1_score(y_test, weighted_prediction, average='weighted')
    rec_results[i] = recall_score(y_test, weighted_prediction, average='weighted')
    pre_results[i] = precision_score(y_test, weighted_prediction, average='weighted')
    print(clf_score)

rmean[0] = results.mean()     # copy mean for 30 iter.
rstd[0] = results.std()       # copy std for 30 iter.
pre_mean[0] = pre_results.mean()  # copy mean for 30 iter.
rec_mean[0] = rec_results.mean()  # copy mean for 30 iter.
f1_mean[0] = f1_results.mean()

comReport[0] = rmean[0]
comReport[1] = rstd[0]
comReport[2] = pre_mean[0]
comReport[3] = rec_mean[0]
comReport[4] = f1_mean[0]
numpy.savetxt("elec2/means/SGDCompleteReportACC_STD_PRE_REC_F1.csv", comReport, fmt='%.5f') # Averages
