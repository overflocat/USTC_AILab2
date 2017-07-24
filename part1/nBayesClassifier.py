from scipy import sparse
import numpy as np

def nBayesClassifier(trainData, trainLabel, testData, testLabel, threshold):
    trainData = trainData.astype(np.bool).astype(np.int32)
    trainLabelP = np.where(trainLabel == 1, 1, 0)
    trainLabelN = np.where(trainLabel == -1, 1, 0)

    pCount = np.sum(trainLabelP)
    nCount = np.sum(trainLabelN)

    trainResultP = sparse.csc_matrix(trainLabelP).dot(trainData).todense()
    trainResultP = trainResultP.astype(np.float)
    trainResultN = sparse.csc_matrix(trainLabelN).dot(trainData).todense()
    trainResultN = trainResultN.astype(np.float)
    trainResultAll = trainResultP + trainResultN
    trainResultAll = trainResultAll / (pCount + nCount)
    trainResultAll[np.isnan(trainResultAll)] = 0.5
    trainResultAll[np.isinf(trainResultAll)] = 0.5

    trainResultP = trainResultP / pCount
    trainResultP[np.isnan(trainResultP)] = 0.5
    trainResultP[np.isinf(trainResultP)] = 0.5

    trainResultN = trainResultN / nCount
    trainResultN[np.isnan(trainResultN)] = 0.5
    trainResultN[np.isinf(trainResultN)] = 0.5

    testData = testData.astype(np.bool).astype(np.int32)
    tempP = testData.multiply(trainResultP).tocsr()
    tempN = testData.multiply(trainResultN).tocsr()
    tempAll = testData.multiply(trainResultAll).tocsr()
    testResult = np.zeros([1, testData.get_shape()[0]])

    for i in range(testData.get_shape()[0]):
        tempM1 = tempP.getrow(i).toarray()
        tempM2 = tempN.getrow(i).toarray()
        tempM3 = tempAll.getrow(i).toarray()

        tempM1 = np.multiply(tempM1, 1 / tempM3)
        tempM2 = np.multiply(tempM2, 1 / tempM3)
        a = np.prod(tempM1[~np.isnan(tempM1)]) * (float(pCount) / (pCount + nCount))
        b = np.prod(tempM2[~np.isnan(tempM2)]) * (float(nCount) / (pCount + nCount))
        testResult[0, i] = a / (a + b)

    testResult = np.asarray(np.where(testResult > threshold, 1, -1)).astype(np.int32)
    accu = np.sum(np.where(testResult == testLabel, 1, 0)) / float(testLabel.shape[0])

    return testResult, accu















