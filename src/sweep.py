
import os
import sys
import numpy as np
import csv
import matplotlib.pyplot as plt

from math import*
from operator import itemgetter
from sklearn import svm
from sklearn import linear_model
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import argparse


path = '../files'

def writeResults(row):
    outFile_writer.writerows(row)

# this utility prints out the corresponding prefix sequence of the predicted best adders
def findPrefixSeq(exMatTagAll, outputList, option, num):
    prefix = []
    if 0 == option:
        prefixAreaPath = os.path.join(path,'prefixAreaDelay' + str(num) + '.csv')
        prefixSeq = open(prefixAreaPath,'w')
    elif 1 == option:
        prefixPowerPath = os.path.join(path,'prefixPowerDelay' + str(num) + '.csv')
        prefixSeq = open(prefixPowerPath,'w')
    elif 2 == option:
        prefixDelayPath = os.path.join(path,'prefixAreaTns.csv')
        prefixSeq = open(prefixDelayPath,'w')
    elif 3 == option:
        prefixTNSPath = os.path.join(path,'prefixPowerTns.csv')
        prefixSeq = open(prefixTNSPath,'w')

    elif 4 == option:
        prefixPPA = os.path.join(path,'prefixPPA' + str(num) + '.csv')
        prefixSeq = open(prefixPPA,'w')

    prefixSeqWriter = csv.writer(prefixSeq)
    for i in outputList:
        for j in range(exMatTagAll.shape[0]):
            if i == int(exMatTagAll[j][0]) :
                prefix.append(exMatTagAll[j][1:])  # append the prefix sequence and targetDelay number
    prefixSeqWriter.writerows(prefix)
    prefixSeq.close()

def sweep(args):
    for num in range(args.repeat):
        selectRatio = 0.3
        
        trainRatio = 0.7
        
        outColNum = 4
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        # need to print out the predicted prefix adder sequence
        needSeq = True
        
        if not args.data_file:
            print("data file load failure.")
            sys.exit(1)
        
        #outPath = os.path.join(path, 'results.txt')
        #outputFile = open(outPath, 'w')
        
        outPath = os.path.join(path, 'selectedID.csv')
        outputFile = open(outPath, 'w')
        IDFileWriter = csv.writer(outputFile)
        
        predictedPath = os.path.join(path, 'predicted_space.csv')
        predicted_space = open(predictedPath, 'w')
        predictedResult = csv.writer(predicted_space)
    
        finalAdderNum = 10
        
        # overall data matrix
        exMatTagAll = np.genfromtxt(args.data_file, delimiter=',', dtype='object')
        #matAll = np.genfromtxt(args.data_file, delimiter=',', dtype='float')
        matAll = exMatTagAll[:,1:] #remove index
        robust_scaler = RobustScaler()
        
        
        #exMatAll = np.genfromtxt(exfile, delimiter=',', dtype='float')
        #exFeatureAll = robust_scaler.fit_transform(exMatAll[:,1:])
        exFeatureAll = robust_scaler.fit_transform(matAll[:,1:-4])
        exFeature = exFeatureAll[:, :2]
        exFeature = np.insert(exFeature, exFeature.shape[1], exFeatureAll[:, -1].transpose(), axis = 1)
        
        exFeature_delay = exFeatureAll[:, :2]
        exFeature_delay = np.insert(exFeature_delay, exFeature_delay.shape[1], exFeatureAll[:, 35:].transpose(), axis = 1)
        # orig exhaustive file, with tag, used for final prefix sequence dumping
        #exfileTag = '../data/tag_4k.csv'
        #if not exfileTag:
        #    print("exhaustiveOrig data file load failure.")
        #    sys.exit(1)

        np.random.seed(num)
        np.random.shuffle(matAll)
        
        matAll = matAll[:int(matAll.shape[0]*selectRatio),:]
        
        mat = robust_scaler.fit_transform(matAll[:,2:])
        
        # remove outlier data
        #featureTrain = mat[:int(mat.shape[0]*trainRatio), :featureNum - 1] # not include slack value
        featureTrain = mat[:int(mat.shape[0]*trainRatio),  : 2] # not include slack value
        featureTrain = np.insert(featureTrain,featureTrain.shape[1], mat[:int(mat.shape[0]*trainRatio), -5].transpose(), axis = 1)  # insert slack value to the last column
        print('Size of selected feature:{}'.format(featureTrain.shape))
        outputTrain = mat[:int(mat.shape[0]*trainRatio), -outColNum: ]
        
        featureTrain_delay = mat[:int(mat.shape[0]*trainRatio),  : 2] 
        featureTrain_delay = np.insert(featureTrain_delay,featureTrain_delay.shape[1], mat[:int(mat.shape[0]*trainRatio), 35:67].transpose(), axis = 1)  # insert slack value to the last column
        
        #featureTest = mat[int(mat.shape[0]*trainRatio):, :featureNum - 1]
        featureTest = mat[int(mat.shape[0]*trainRatio):, : 2]
        featureTest = np.insert(featureTest, featureTest.shape[1], mat[int(mat.shape[0]*trainRatio):, -5].transpose(), axis = 1)  # insert slack value to the last column`
        outputTest = mat[int(mat.shape[0]*trainRatio):, -outColNum: ]
        
        
        featureTest_delay = mat[int(mat.shape[0]*trainRatio):, : 2]
        featureTest_delay = np.insert(featureTest_delay, featureTest_delay.shape[1], mat[int(mat.shape[0]*trainRatio):, 35:67].transpose(), axis = 1)  # insert slack value to the last column`
        
        #iteration = [1000, 0, 1, 10, 0.1, 100, 0.01, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 20, 30, 40, 50, 60]
        iteration = [1000, 0, 1, 10, 0.1, 100, 0.01, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 20, 30, 40, 50]

        # different regression models
        regr_1 = svm.SVR(kernel='rbf', C=1.3, gamma=0.5)
        #regr_1 = svm.SVR(kernel='rbf', C=1.3, gamma=0.5, epsilon=0.01)
        regr_2 = svm.SVR(kernel='poly', C=1e3, gamma=0.1)
        regr_3 = svm.SVR(kernel='linear', C=2e3, gamma=0.3)
        
        # split each output's model, so that we can tune more flexible
        regrOutput0 = regr_1
        regrOutput1 = regr_1
        regrOutput2 = regr_3
        regrOutput3 = regr_3
        
        
        print("area-delay iteration: ")
        overall0 = np.array([])
        
        iteration_alpha = [1000, 0, 1, 10, 0.1, 100, 0.01, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 20, 30, 40, 50, 60]
        iteration_beta = [1000, 0, 1, 10, 0.1, 100, 0.01, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 20, 30, 40, 50, 60]
    
        if args.dim == 2:
    
            for i in iteration:
                areaDelayMatTrain = i * outputTrain[:,0] + outputTrain[:,2]
                areaDelayMatTest = i * outputTest[:,0] + outputTest[:,2]
                regrM0 = regrOutput0.fit(featureTrain_delay, areaDelayMatTrain)
                result0 = regrM0.predict(featureTest_delay)
                print(" ")
                print(i, " result0 mse: ", mean_squared_error(result0, areaDelayMatTest))
                print("result0 r2 ", r2_score(result0, areaDelayMatTest))
            
                output_0 = regrM0.predict(exFeature_delay)
                outputList_0 = output_0.argsort()[:finalAdderNum]
                outputData = np.insert(exFeatureAll, exFeatureAll.shape[1], output_0, axis=1)
                overall0 = np.append(overall0, outputList_0)
                output_0Sorted =sorted( output_0.tolist())
            overall0 = np.array(list(set(overall0.tolist())))
            print(overall0.shape)
            print(overall0)
            IDFileWriter.writerow(overall0.tolist())
            predictedResult.writerow(output_0)
            if needSeq:
                findPrefixSeq(exMatTagAll, overall0, 0, num)
            outputFile.close()
    
            print("power-delay iteration: ")
            overall1 = np.array([])
            
            for i in iteration:
                powerDelayMatTrain = i * outputTrain[:,1] + outputTrain[:,2]
                powerDelayMatTest = i * outputTest[:,1] + outputTest[:,2]
                regrM1 = regrOutput1.fit(featureTrain_delay, powerDelayMatTrain)
                result1 = regrM1.predict(featureTest_delay)
                print(" ")
                print(i, " result1 mse: ", mean_squared_error(result1, powerDelayMatTest))
                print("result1 r2 ", r2_score(result1, powerDelayMatTest))
            
                output_1 = regrM1.predict(exFeature_delay)
                outputList_1 = output_1.argsort()[:finalAdderNum]
                print(outputList_1)
                outputData = np.insert(exFeatureAll, exFeatureAll.shape[1], output_1, axis=1)
                overall1 = np.append(overall1, outputList_1)
            
            overall1 = np.array(list(set(overall1.tolist())))
            print(overall1.shape)
            if needSeq:
                findPrefixSeq(exMatTagAll, overall1, 1, num)
        if args.dim == 3:
            for i in iteration_alpha:
                for j in iteration_beta:
                    ppaMatTrain = i * outputTrain[:, 0] + j * outputTrain[:, 1] + outputTrain[:, 2]
                    ppaMatTest = i * outputTest[:, 0] + j * outputTest[:, 1] + outputTest[:, 2]
                    regrM0 = regrOutput0.fit(featureTrain_delay, ppaMatTrain)
                    result0 = regrM0.predict(featureTest_delay)
                    print(" ")
                    print(i, " result0 mse: ", mean_squared_error(result0, ppaMatTest))
                    print("result0 r2 ", r2_score(result0, ppaMatTest))
            
                    output_0 = regrM0.predict(exFeature_delay)
                    outputList_0 = output_0.argsort()[:finalAdderNum]
                    print(outputList_0)
                    outputData = np.insert(exFeatureAll, exFeatureAll.shape[1], output_0, axis=1)
                    overall0 = np.append(overall0, outputList_0)
                    output_0Sorted =sorted( output_0.tolist())
            overall0 = np.array(list(set(overall0.tolist())))
            print(overall0.shape)
            print(overall0)
            IDFileWriter.writerow(overall0.tolist())
            predictedResult.writerow(output_0)
            if needSeq:
                findPrefixSeq(exMatTagAll, overall0, 4, num)
            outputFile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type = int, default = 2, help = 'dimension of search space')
    parser.add_argument('--repeat', type = int, default = 1, help = 'number of repeated experiments')
    parser.add_argument('--data_file', type = str, default = '../data/feature_values.csv', help = 'exhaustive solutions')
    
    args = parser.parse_args()
    
    sweep(args)

