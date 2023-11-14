import numpy as np
import algorithms.batch.QRNeuralNetwork
import datetime
import tensorflow as tf
import algorithms.evaluation.coverage
import csv

def runACI(output, input, alpha, alpha_range, step, tinit, splitSize):
    T = len(output)    
    alpha = alpha.T
    alphaTrajectory = np.zeros(T - tinit)
    adaptErrSeq = np.zeros(T - tinit)
    hozon = np.zeros(T - tinit)
    alphat = alpha_range
    
    for t in range(tinit, T):
        trainPoints = np.random.choice(np.arange(t - 1), size=int(splitSize * (t - 1)), replace=False)
        calpoints = np.delete(np.arange(t - 1), trainPoints)
        Xtrain = input[trainPoints, :]
        Ytrain = output[trainPoints]
        XCal = input[calpoints, :]
        YCal = output[calpoints]

        # Fit quantile regression on the training set
        
        QR = algorithms.batch.QRNeuralNetwork.QRNN(alpha=alpha, input_train=Xtrain, output_train=Ytrain)
        QR.__init__(alpha=alpha, input_train=Xtrain, output_train=Ytrain)
        QR.pre_learning()
        low = QR.predict(input_test=XCal)[0]
        high = QR.predict(input_test=XCal)[1]
        lower = low.reshape(-1, 1)
        higher = high.reshape(-1, 1)        
        scores = np.maximum(YCal - higher, lower - YCal)
        #コンフォーマル予測
        confQuantAdapt = np.percentile(scores, alphat * 100)
        X = np.full([len(scores), 1], confQuantAdapt)
        higher = higher + X.reshape(-1, 1)
        lower = lower - X.reshape(-1, 1)
        scores = np.maximum(YCal - higher, lower - YCal)

        #lqrfitUpper = model.fit(q=1 - alpa / 2)
        #lqrfitLower = model.fit(q=alpa / 2)
        #Up_Array=np.vstack((lqrfitUpper.params, lqrfitUpper.params))
        #Low_Array=np.vstack((lqrfitLower.params, lqrfitLower.params))
        #onesA = np.array(np.ones(XCal.shape[0]))
        #ones = onesA.reshape(-1, 1)

        # Compute conformity score on the calibration set
        #predLowForCal = np.dot(np.hstack((ones, XCal)), Up_Array)
        #predUpForCal = np.dot(np.hstack((ones, XCal)), Low_Array)
        #scores = np.maximum(YCal - higher, lower - YCal)

        low = QR.predict([input[t]])[0]
        high = QR.predict([input[t]])[1]        
        lower = low.reshape(-1, 1) - confQuantAdapt
        higher = high.reshape(-1, 1) + confQuantAdapt
        newScore = max(output[t] - higher, lower - output[t])
    
        if alphat >= 1:
            adaptErrSeq[t - tinit] = 0
        elif alphat <= 0:
            adaptErrSeq[t - tinit] = 1
        else:
            confQuantAdapt = np.percentile(scores, alphat * 100)
            adaptErrSeq[t - tinit] = int(confQuantAdapt < newScore)

        alphaTrajectory[t - tinit] = alphat
        
        alphat += step * (adaptErrSeq[t - tinit] - 1 + alpha_range)

        now = datetime.datetime.now()

        covlen = (np.where((scores <= 0), 1, 0))
        print(t)
        print(covlen)
        Covrate = (np.sum(covlen) + int(newScore <= 0))/(t - len(trainPoints))
        with open('log2.txt', 'a') as f:
            f.write('\n' + '\t' + str(t) + ' / ' + str(T) + ' : ' + str(now))
            f.write('\n' + '\t\tCoverage rate = ' + str(Covrate))
            hozon[t - tinit] = str(Covrate) 

            tf.print(f"Done {t} time steps")

    with open('sample_writer_row.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        for i in hozon:
            writer.writerow([i])

    return alphaTrajectory, adaptErrSeq