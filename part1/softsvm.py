from sklearn.svm import SVC
import numpy as np

def softsvm(trainData, trainLabel, testData, testLabel, sigma, Ci):
    if sigma == 0:
        clf = SVC(C=Ci, kernel='linear')
    else:
        clf = SVC(C=Ci, kernel='rbf', gamma=sigma)
    clf.fit(trainData, trainLabel)

    testResult = np.array(clf.predict(testData))
    testResult = np.where(testResult > 0, 1, -1).astype(np.int32)
    accu = np.sum(np.where(testResult == testLabel, 1, 0)) / float(testLabel.shape[0])

    return testResult, accu