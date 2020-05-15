import numpy
import pandas
import time
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, recall_score, precision_score

def conceptdriftrate(arr,nb,niter):   # parameters (array , num batches , num iteratoins )
    condriftmeans = numpy.zeros(shape=(niter))
    driftcounter = 0
    templi = numpy.zeros(shape=(nb))
    counter = 0
    for c in range(30):
        col = arr[ :,c]
        for el in range(len(col)):
            if el + 1 not in range(len(col)):
                break
            else:
                ans = abs(arr[el + 1, c] - arr[el, c])
                templi[counter] = ans
                counter = counter + 1
        #print(templi.mean())
        condriftmeans[driftcounter] = templi.mean()
        driftcounter = driftcounter + 1
        counter = 0
    return condriftmeans


df=pandas.read_csv( 'datasets/shuttle.csv') # Read data from CSV
array=df.values                                     # Convert to array

X=array[:,0:9]                                     # Seperate classifying features
Y=array[:,9]                                       # Seperate target
scaler=MinMaxScaler(feature_range=(0,1))            # Decale normilizing method Min/Max
rX=scaler.fit_transform(X)                          # Normalize X array
exp_nums = [150]#[50, 100, 150, 200, 300, 400, 500, 1000] # The chunksizes that are going to be examined in the experiment
exp_num_counter = 0                                 # this will shift the total averages according to the exp elements
rmean = numpy.zeros(shape=(1))                      # get mean results for one chunk size
rstd = numpy.zeros(shape=(1))                       # get std results for one chunk size
pre_mean = numpy.zeros(shape=(1))                            # get mean results for one chunk size
rec_mean = numpy.zeros(shape=(1))                            # get mean results for one chunk size
f1_mean = numpy.zeros(shape=(1))                             # get mean results for one chunk size
timesmean = numpy.zeros(shape=(1))                   # get mean of all times for one chunk size
timesmeans = numpy.zeros(shape=(len(exp_nums)))                   # get mean of all times for one chunk size
rmeans = numpy.zeros(shape=(len(exp_nums)))                     # put all means in one array
rstds = numpy.zeros(shape=(len(exp_nums)))                      # put all stds in one array
pre_means = numpy.zeros(shape=(len(exp_nums)))                     # put all means in one array
rec_means = numpy.zeros(shape=(len(exp_nums)))                     # put all means in one array
f1_means = numpy.zeros(shape=(len(exp_nums)))                     # put all means in one array
CDRmeans = numpy.zeros(shape=(len(exp_nums)))
times = numpy.zeros(shape=(30))                     # Define array to store times to execute incremental learning iterations

clf = SGDClassifier(alpha=0.0001, average=False, class_weight=None,
              early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
              l1_ratio=0.15, learning_rate='optimal', loss='modified_huber',
              max_iter=1000, n_iter_no_change=5, n_jobs=None, penalty='none',
              power_t=0.5, random_state=None, shuffle=True, tol=0.001,
              validation_fraction=0.1, verbose=0, warm_start=False)

concept_drift_rate = numpy.zeros(shape=(30))
for e in exp_nums:
    n_iter = 30         # Number of iterations
    n = e               # Chunksize
    exp_rows=int(((len(X)*0.7))/n)
    results = numpy.zeros(shape=(exp_rows, n_iter))
    pre_results = numpy.zeros(shape=(exp_rows, n_iter))
    rec_results = numpy.zeros(shape=(exp_rows, n_iter))
    f1_results = numpy.zeros(shape=(exp_rows, n_iter))
    co=0                                            # column number: used to change  the column index when inserting data to results matrix

    for i in range(n_iter):
        x_train, x_test, y_train, y_test = train_test_split(rX, Y, test_size=0.3)  # splitting data

        Xbatches = [x_train[i:i + n] for i in range(0, x_train.shape[0], n)]  # access them by list_df[0] etc. [7]
        Ybatches = [y_train[i:i + n] for i in range(0, y_train.shape[0], n)]  # access them by list_df[0] etc. [7]

        num_rows = len(x_train)
        num_batches = int(num_rows / n)
        start_time = int(round(time.time() * 1000))                             # to start counting before iteration
        for b in range(num_batches):                                            # this range is related to the number of splits
            clf.partial_fit(Xbatches[b], Ybatches[b], classes=numpy.unique(Y))  # fit the given batch
            clf_score=clf.score(x_test, y_test)                                 # test the current model with the test data
            results[b,co]=clf_score                 #

            weighted_prediction = clf.predict(x_test)
            f1_results[b, co] = f1_score(y_test, weighted_prediction, average='weighted')
            rec_results[b, co] = recall_score(y_test, weighted_prediction, average='weighted')
            pre_results[b, co] = precision_score(y_test, weighted_prediction, average='weighted')

            numpy.savetxt("shuttle/results/SGDResult" + str(n) + ".csv",  results, fmt='%.5f', delimiter=';', newline='\n') # export results file
            print(clf_score)
        end_time = int(round(time.time() * 1000))   # to start counting before iteration
        co += 1                                     # shift to next column
        elapsed_time = end_time - start_time        # calculate time taken
        times[i]=elapsed_time                       # add time for iteration to times array
        numpy.savetxt("shuttle/times/SGDTT" + str(n) + ".csv", times, fmt='%.5f')    # export file

    timesmean[0] = times.mean()
    numpy.savetxt("times/SGDTMEAN" + str(n) + ".csv", times, fmt='%.5f')  # export file
    rmean[0] = results[num_batches - 1,].mean()     # copy mean for 30 iter. for on chunk size
    rstd[0] = results[num_batches - 1,].std()       # copy std for 30 iter. for one chunk size
    pre_mean[0] = pre_results[num_batches - 1,].mean()  # copy mean for 30 iter. for on chunk size
    rec_mean[0] = rec_results[num_batches - 1,].mean()  # copy mean for 30 iter. for on chunk size
    f1_mean[0] = f1_results[num_batches - 1,].mean()  # copy mean for 30 iter. for on chunk size

    numpy.savetxt("shuttle/results/SGDAVG" + str(n) + ".csv", rmean, fmt='%.5f')
    numpy.savetxt("shuttle/results/SGDSTD" + str(n) + ".csv", rstd, fmt='%.5f')
    rmeans[exp_num_counter] = results[num_batches - 1,].mean()  # copy mean for 30 iter. for on chunk size for current
    rstds[exp_num_counter] = results[num_batches - 1,].std()  # copy std for 30 iter. for one chunk size for current
    pre_means[exp_num_counter] = pre_results[num_batches - 1,].mean()  # copy mean for 30 iter. for on chunk size for current
    rec_means[exp_num_counter] = rec_results[num_batches - 1,].mean()  # copy mean for 30 iter. for on chunk size for current
    f1_means[exp_num_counter] = f1_results[num_batches - 1,].mean()  # copy mean for 30 iter. for on chunk size for current

    timesmeans[exp_num_counter] = times.mean()  # copy timeAVG for 30 iter. for one chunk size for current

    concept_drift_rate = conceptdriftrate(results, num_batches, 30)
    numpy.savetxt("shuttle/means/CONDRIFTRATE.csv", concept_drift_rate, fmt='%.5f')
    print(concept_drift_rate)
    print(concept_drift_rate.mean())
    CDRmeans[exp_num_counter] = concept_drift_rate.mean()

    exp_num_counter = exp_num_counter +1 # shift all mean arrays

numpy.savetxt("shuttle/means/SGDAVGALLEXP.csv", rmeans, fmt='%.5f') # Averages
numpy.savetxt("shuttle/means/SGDSTDALLEXP.csv", rstds, fmt='%.5f') # Standard Deviation
numpy.savetxt("shuttle/means/SGDTIMESALLEXP.csv", timesmeans, fmt='%.5f') # Times
numpy.savetxt("shuttle/means/SGDCDRALLEXP.csv", CDRmeans, fmt='%.5f') # Concept drift rate
numpy.savetxt("shuttle/means/SGDPRECISION.csv", pre_means, fmt='%.5f') # Averages
numpy.savetxt("shuttle/means/SGDRECALL.csv", rec_means, fmt='%.5f') # Averages
numpy.savetxt("shuttle/means/SGDF1.csv", f1_means, fmt='%.5f') # Averages



