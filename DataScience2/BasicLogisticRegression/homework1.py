import argparse
import numpy as np
import pandas as pd
import math

STEP_SIZE = 0.0001

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Homework 1",
        epilog = "CSCI 4360/6360 Data Science II: Fall 2023",
        add_help = "How to use",
        prog = "python homework1.py [train-data] [train-label] [test-data]")
    parser.add_argument("paths", nargs = 3)
    args = vars(parser.parse_args())

#     Access command-line arguments as such:
    
rtrainData = args["paths"][0]
rtrainLabels = args["paths"][1]
rtestData = args["paths"][2]

rawtrainData = np.loadtxt(rtrainData, delimiter = ' ')
nptrainLabels = np.loadtxt(rtrainLabels, delimiter = ' ')
rawtestData = np.loadtxt(rtestData, delimiter = ' ')

#---transforming the data---#

trainsetWords = set(rawtrainData[:,1])
trainWords = np.array(sorted(trainsetWords))
trainsetDocs = set(rawtrainData[:,0])
trainDocs = np.array(sorted(trainsetDocs))

pivtrainData = np.zeros((len(trainDocs), len(trainWords)))

for x in rawtrainData:
    pivtrainData[int(np.argwhere(trainDocs == x[0])), int(np.argwhere(trainWords == x[1]))] = x[2]
    

testsetWords = set(rawtestData[:,1])
testWords = np.array(sorted(testsetWords))
testsetDocs = set(rawtestData[:,0])
testDocs = np.array(sorted(testsetDocs))

pivtestData = np.zeros((len(testDocs), len(testWords)))

for x in rawtestData:
    pivtestData[int(np.argwhere(testDocs == x[0])), int(np.argwhere(testWords == x[1]))] = x[2]

#---#---#

#---cleaning the data---#

for word in reversed(trainWords):
    if word not in testWords:
        pivtrainData = np.delete(pivtrainData, int(np.argwhere(trainWords == word)), axis = 1)
        
        
for word in reversed(testWords):
    if word not in trainWords:
        pivtestData = np.delete(pivtestData, int(np.argwhere(testWords == word)), axis = 1)
        
#---#---#

def gradient(X,y,w):
    u=np.zeros(X.shape[0])
    i=0
    for x in X:
        #print(math.exp(w@x)/(math.exp(w@x)+1))
        u[i] = (math.exp(w@x)/(math.exp(w@x)+1))
        #print(u[i])
        i=i+1
    #print(u)
    g = X.T@(u - y)
    return g

def LR(w,n,X,y,maxIt):
    i=0
    
    while ((i < maxIt)):
        g = n*gradient(X,y,w)
        #print(g[1])
        nw = w - g
        w = nw
        #print(w[0])
        #print(max(w))
        i = i + 1
    return w





w = np.zeros(pivtrainData.shape[1])
#print(w.shape)
n = STEP_SIZE
#print("FINAL")
w = LR(w,n,pivtrainData,nptrainLabels,10000)
#print(w)


#prediction of test data
i=0
for x in pivtestData:
    Py = math.exp(x@w)/(math.exp(x@w)+1)
    if Py > 0.5:
        print(1)
        
    else:
        print(0)
