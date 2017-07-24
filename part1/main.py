from scipy import sparse
import numpy as np
from getFeature import getFeature
from getFeature import getFeatureFromModel
from softsvm import softsvm
from lsClassifier import lsClassifier
from nBayesClassifier import nBayesClassifier

featureM, evalRes = getFeature()

threshold = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
shapeM = featureM.get_shape()
sliceL = len(evalRes) / 5
nBayesResult = np.zeros([len(threshold), 5])
for index in range(len(threshold)):
    for i in range(5):
        trainData = sparse.vstack([featureM[0:i*sliceL, :], featureM[(i+1)*sliceL:shapeM[0], :]])
        trainLabel = evalRes[0:i*sliceL] + evalRes[(i+1)*sliceL:shapeM[0]]
        testData = featureM[i*sliceL:(i+1)*sliceL, :]
        testLabel = evalRes[i*sliceL:(i+1)*sliceL]
        testResult, accu = nBayesClassifier(trainData, np.array(trainLabel), testData, np.array(testLabel), threshold[index])
        nBayesResult[index, i] = accu
        print threshold[index], i, accu

print nBayesResult

lambdaS = [0.0001, 0.01, 0.1, 0.5, 1, 5, 10, 100, 1000, 5000, 10000]
shapeM = featureM.get_shape()
sliceL = len(evalRes) / 5
lsResult = np.zeros([len(lambdaS), 5])
for index in range(len(lambdaS)):
    for i in range(5):
        trainData = sparse.vstack([featureM[0:i*sliceL, :], featureM[(i+1)*sliceL:shapeM[0], :]])
        trainLabel = evalRes[0:i*sliceL] + evalRes[(i+1)*sliceL:shapeM[0]]
        testData = featureM[i*sliceL:(i+1)*sliceL, :]
        testLabel = evalRes[i*sliceL:(i+1)*sliceL]
        testResult, accu = lsClassifier(trainData, np.array(trainLabel), testData, np.array(testLabel), lambdaS[index])
        lsResult[index, i] = accu
        print lambdaS[index], i, accu

print lsResult

featureM, evalRes = getFeatureFromModel()
Ci = [1, 10, 100, 1000]
shapeM = featureM.shape
dSum = 0
for i in range(shapeM[0] / 1000):
    for j in range(shapeM[0] / 1000):
        temp = featureM[i, :] - featureM[j, :]
        dSum += np.sum(temp * temp)
d = dSum / ((shapeM[0] / 1000) * (shapeM[0] / 1000))
sigmaL = [0, 0.01 * d, 0.1 * d, d, 10 * d, 100 * d]
sliceL = len(evalRes) / 5
svmResult = np.zeros([len(Ci), 5 * len(sigmaL)])
for indexC in range(len(Ci)):
    for indexD in range(len(sigmaL)):
        for i in range(5):
            trainData = np.vstack([featureM[0:i*sliceL, :], featureM[(i+1)*sliceL:shapeM[0], :]])
            trainLabel = evalRes[0:i*sliceL] + evalRes[(i+1)*sliceL:shapeM[0]]
            testData = featureM[i*sliceL:(i+1)*sliceL, :]
            testLabel = evalRes[i*sliceL:(i+1)*sliceL]
            testResult, accu = softsvm(trainData, np.array(trainLabel), testData, np.array(testLabel), sigmaL[indexD], Ci[indexC])
            svmResult[indexC, i*indexD] = accu
            print sigmaL[indexD], Ci[indexC], i, accu

print svmResult
