from PIL import Image as im
from PIL import ImageDraw
import numpy as np
import numpy.matlib

def nearestDistance(pointMat, clusterCenters):
    disResult = np.mat(np.zeros([np.shape(pointMat)[0], np.shape(clusterCenters)[0]]))
    for i in range(np.shape(clusterCenters)[0]):
        temp = pointMat - np.matlib.repmat(clusterCenters[i, :], np.shape(pointMat)[0], 1)
        disResult[:, i] = np.sum(np.multiply(temp, temp), axis=1)
    disMinIndex = np.mat(np.argmin(disResult, axis=1))
    return disMinIndex

def initializeClusterCenters(pointMat, k):
    clusterCenters = np.zeros([k, 3])
    for i in range(k):
        clusterCenters[i, :] = pointMat[np.random.randint(0, np.shape(pointMat)[0]), :]

    return clusterCenters

def kmeans(pointMat, k, clusterCenters):
    subCenter = np.mat(np.zeros([np.shape(pointMat)[0], 1], dtype=np.int32))
    for count in range(50):
        disMinIndex = nearestDistance(pointMat, clusterCenters)
        subCenter = disMinIndex
        for i in range(k):
            clusterCenters[i, :] = np.mean(pointMat[np.where(subCenter == i)[0], :], axis=0)

    return subCenter, clusterCenters

k = 3
fp = open("sea.JPG", "rb")
image = im.open(fp)
imageData = np.mat(image.getdata())
clusterCenters = initializeClusterCenters(imageData, k)
subCenter, clusterCenters = kmeans(imageData, k, clusterCenters)

clusterCenters = clusterCenters.astype(np.int32)

m, n = image.size
draw = ImageDraw.Draw(image)
for i in range(m*n):
    draw.point((i % m, i / m), tuple(clusterCenters[int(subCenter[i])]))
del draw

image.show()
