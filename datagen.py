#!/usr/bin/python
import random
import numpy as np

def generateDist(size=100, skew=1.5, error_rate=0.10):
    dist = np.random.zipf(skew, size)
    
    for i in range(0,size):
        if np.random.rand(1,1) < 1-error_rate:
            dist[i] = 0

    dist = dist / float(np.max(dist))

    return dist

def generateDataset(dist, workers=30):
    n = np.size(dist)
    z = np.zeros((n,workers))
    for i in range(0, workers):
        for j in range(0,n):
            if np.random.rand(1,1) < dist[j]:
                z[j,i] = 1

    return z

def generateWeightedDataset(dist, workers=30, shuffle=0):
    n = np.size(dist)
    z = np.zeros((n,workers))
    for i in range(0, workers):
        for j in range(0,n):
            if np.random.rand(1,1) < dist[j]:
                z[j,i] = 1

    d = [i[0] for i in sorted(enumerate(dist), key=lambda x:x[1])]
    
    d = shuffleList(d,shuffle)

    return (z, d)

def shuffleList(l, shuffle):
    N = len(l)
    for i in range(0,shuffle):
        index1 = int(np.random.rand(1,1)*N)
        index2 = int(np.random.rand(1,1)*N)
        tmp = l[index1]
        l[index1] = l[index2]
        l[index2] = tmp

    return l

