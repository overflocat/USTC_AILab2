from scipy.sparse import csr_matrix
import numpy as np
import random
import re
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import gensim
import glob
import string

def getFeature():
    fileData = open("data")
    row = []
    col = []
    data = []
    evalRes = []
    rowIndex = -1
    fileList = fileData.readlines()
    random.shuffle(fileList)
    for line in fileList:
        line = line.rstrip('\n')
        dataList = re.split(' |:', line)

        if int(dataList[0]) >= 7:
            evalRes.append(1)
        else:
            if int(dataList[0]) <= 4:
                evalRes.append(-1)
            else:
                continue
        del dataList[0]

        rowIndex = rowIndex + 1
        row.extend([rowIndex] * int(len(dataList) / 2))
        col.extend(map(int, dataList[::2]))
        data.extend(map(int, dataList[1::2]))

    featureMatrix = csr_matrix((data, (row, col)))
    featureMNew = SelectKBest(chi2, k=20000).fit_transform(featureMatrix, evalRes)
    return featureMNew, evalRes

def getFeatureFromModel():
    model = gensim.models.Word2Vec.load('med150.model.bin')
    documents = []
    evalRes = []
    txtNames = glob.glob("original/*.txt")
    for fileName in txtNames:
        fp = open(fileName)
        temp = re.split('_|\.', fileName)
        if int(temp[1]) >= 7:
            evalRes.append(1)
        else:
            evalRes.append(-1)
        buf = fp.readline()
        documents.append(buf)

    stoplist = set('for a of the and to in at'.split())
    texts = [[word for word in document.translate(string.maketrans("", ""), string.punctuation).lower().split() if
              word not in stoplist]
             for document in documents]

    featureMatrix = np.zeros([len(texts), 150])
    i = -1
    for line in texts:
        i = i + 1
        count = 0
        for word in line:
            try:
                featureMatrix[i, :] = featureMatrix[i, :] + np.array(model[word])
            except KeyError:
                count += 1
        featureMatrix[i, :] /= (len(line) - count)

    return featureMatrix, evalRes

