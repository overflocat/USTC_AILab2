from sklearn import linear_model
import numpy as np

def lsClassifier(trainData, trainLabel, testData, testLabel, lambdaS):
    reg = linear_model.Ridge(alpha=lambdaS)
    reg.fit(trainData, trainLabel.tolist())

    W = reg.coef_
    testResult = np.array(testData.dot(W))
    testResult = np.where(testResult > 0, 1, -1).astype(np.int32)
    accu = np.sum(np.where(testResult == testLabel, 1, 0)) / float(testLabel.shape[0])

    return testResult, accu
