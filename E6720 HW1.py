import numpy as np
import math
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import nbinom
import seaborn as sns


def predict(Xtest,X0,X1,gamma_pars,e,f):

    a,b =  gamma_pars
    n0 = np.shape(X0)[0]
    n1 = np.shape(X1)[0]

    logXpred0 = np.sum(nbinom.logpmf(Xtest, a + np.sum(X0[:,:-1],axis = 0), (n0 + a)/(n0 + b + 1)),axis = 1)
    logXpred1 = np.sum(nbinom.logpmf(Xtest, a + np.sum(X1[:,:-1],axis = 0), (n1 + a)/(n1 + b + 1)),axis = 1)


    y0Haty = logXpred0 + math.log((e + n0)/(n0 + n1 + e + f))
    y1Haty = logXpred1 + math.log((f + n1)/(n0 + n1 + e + f))


    return y0Haty,y1Haty

#calculate the componet of the confusion matrix
def calculateConfusionMtrix(Ytest,predX0,predX1,numX):
    FF = 0
    TT = 0
    FT = 0
    TF = 0
    Xlabel = np.zeros(numX)

    for i in range (numX):
        if predX0[i] < predX1[i]:
            Xlabel[i] = 1
        else:
            Xlabel[i] = 0

    for i in range(numX):
        if Xlabel[i] == Ytest[i] and Ytest[i] == 0:
            FF += 1

        if Xlabel[i] == Ytest[i] and Ytest[i] == 1:
            TT += 1

        if Xlabel[i] != Ytest[i] and Ytest[i] == 0:
            TF += 1

        if Xlabel[i] != Ytest[i] and Ytest[i] == 1:
            FT += 1


    return FF,TT,TF,FT,Xlabel


def confusion_Matrix(FF,FT,TF,TT):
    table = pd.DataFrame([[FF,FT],[TF,TT]])
    table.index.name = ["predict not spam","predct spam"]
    table.columns = ["actually not spam","actually spam"]
    return table

def expected_lemda(gamma_pars,X0,X1):
    a,b = gamma_pars
    n0 = np.shape(X0)[0]
    n1 = np.shape(X1)[0]
    expected_lemda0 = (a + np.sum(X0[:,:-1],axis = 0))/(b + n0)
    expected_lemda1 = (a + np.sum(X1[:,:-1],axis = 0))/(b + n1)
    return expected_lemda0,expected_lemda1

def findErrorPre(Xpre,Ytest,numX):
    errorIdx = np.zeros(0)
    for i in range(numX):
        if Xpre[i] != Ytest[i]:
            errorIdx = np.append(errorIdx,np.array([i]))

    return errorIdx

def find_smallest_3indices(Xpred0,Xpred1,numX):
    Xdiff = np.zeros(numX)
    Xdiff = np.abs(Xpred1 - Xpred0)
    Indices = Xdiff.argsort()[:3]
    return Indices



if __name__ == "__main__":
    #load data
    df_Xtrain = pd.read_csv("X_train.csv")
    df_Xtest = pd.read_csv("X_test.csv")
    df_Ytrain = pd.read_csv("label_train.csv")
    df_Ytest = pd.read_csv("label_test.csv")

    #append label to X
    df_Xtrain['Y'] = df_Ytrain

    #select label X0 and X1
    df_Xtrain0 = df_Xtrain[df_Xtrain['Y'] == 0]
    df_Xtrain1 = df_Xtrain[df_Xtrain['Y'] == 1]

    #put data into array
    Xtrain0 = np.array(df_Xtrain0)
    Xtrain1 = np.array(df_Xtrain1)
    Xtest = np.array(df_Xtest)
    Ytest = np.array(df_Ytest)

    #compute parameters
    gamma_paras = (1,1)
    e = 1
    f = 1
    numX = np.shape(Xtest)[0]

    #initialize prediction array
    predX0 = np.zeros(numX)
    predX1 = np.zeros(numX)

    #make prediction
    predX0, predX1 = predict(Xtest, Xtrain0, Xtrain1, gamma_paras,e,f)


    #b)
    #initialize predction count
    #first index is what the prediction return
    #second index is what the email actually is
    Ytest.reshape(numX,)
    Xlabel = np.zeros(numX)

    FF, TT, TF, FT, Xlabel= calculateConfusionMtrix(Ytest,predX0,predX1,numX)
    table = confusion_Matrix(FF,FT,TF,TT)
    print(table)
    accuracy = (FF + TT)/(numX)
    print("accuracy",accuracy)

    #c)
    #load the coloum name
    df_readMe = pd.read_csv("README")
    df_readMe = df_readMe.iloc[1:]

    #put data in array
    readMe = np.array(df_readMe)

    #calculate expected lemda0 and lemda1
    lemda0,lemda1 = expected_lemda(gamma_paras,Xtrain0, Xtrain1)

    #find the index of error prediction
    errorIdx = findErrorPre(Xlabel,Ytest,numX)
    print("Error Index",errorIdx)

    #pack the items we need in df frame for plotting graph latter
    #useing 1st X, 24th X, 49th X
    readMe = readMe.reshape(54,).tolist()
    Xerror1 = Xtest[1:2:1,::1].reshape(54,).tolist()
    Xerror2 = Xtest[24:25:1,::1].reshape(54,).tolist()
    Xerror3 = Xtest[49:50:1,::1].reshape(54,).tolist()
    df = pd.DataFrame({"lemda0":lemda0,"lemda1":lemda1,"Xerror1":\
        Xerror1,"readMe":readMe,"Xerror2":Xerror2,"Xerror3": Xerror3})

    #plot first error prediction with expected lemda0 and lemda1
    fig1, ax = plt.subplots()
    ax.scatter(np.arange(len(df['readMe'])), df['Xerror1'],label = "Error Prediction1 Feature")
    ax.scatter(np.arange(len(df['readMe'])), df['lemda0'],label = "Expected Frequency Lemda0")
    ax.scatter(np.arange(len(df['readMe'])), df['lemda1'],label = "Expected Frequency Lemda1")
    ax.xaxis.set_ticks(np.arange(len(df['readMe'])))
    ax.xaxis.set_ticklabels(df['readMe'], rotation = 90)
    plt.xlabel("labels")
    plt.ylabel("frequency")
    plt.title("error prediction1 label compare vs expected")
    plt.legend()
    plt.show()

    #plot the second error prediction with expected lemda0 and lemda1
    fig2, ax = plt.subplots()
    ax.scatter(np.arange(len(df['readMe'])), df['Xerror2'],label = "Error Prediction2 Featrue")
    ax.scatter(np.arange(len(df['readMe'])), df['lemda0'],label = "Expected Frequency Lemda0")
    ax.scatter(np.arange(len(df['readMe'])), df['lemda1'],label = "Expected Frequency Lemda1")
    ax.xaxis.set_ticks(np.arange(len(df['readMe'])))
    ax.xaxis.set_ticklabels(df['readMe'], rotation = 90)
    plt.xlabel("labels")
    plt.ylabel("frequency")
    plt.title("error prediction2 label compare vs expected")
    plt.legend()
    plt.show()

    #plot the thrid error prediction with expected lemda0 and lemda1
    fig3, ax = plt.subplots()
    ax.scatter(np.arange(len(df['readMe'])), df['Xerror3'],label ="Error Prediction3 Feature")
    ax.scatter(np.arange(len(df['readMe'])), df['lemda0'],label = "Expected Frequency Lemda0")
    ax.scatter(np.arange(len(df['readMe'])), df['lemda1'],label = "Expected Frequency Lemda1")
    ax.xaxis.set_ticks(np.arange(len(df['readMe'])))
    ax.xaxis.set_ticklabels(df['readMe'], rotation = 90)
    plt.xlabel("labels")
    plt.ylabel("frequency")
    plt.title("error prediction3 label compare vs expected")
    plt.legend()
    plt.show()

    #calculate the prediction probability
    error1PredP = math.exp(predX0[1])/ (math.exp(predX0[1]) + math.exp(predX1[1]))
    error2PredP = math.exp(predX0[24]) / (math.exp(predX0[24]) + math.exp(predX1[24]))
    error3PredP = math.exp(predX0[49]) / (math.exp(predX0[49]) + math.exp(predX1[49]))
    print("nonSpam predict probability", error1PredP,error2PredP,error3PredP)


    #d)
    #find the indeces of the 3 most ambigous prediction
    indces = find_smallest_3indices(predX0,predX1,numX)
    print("most ambigous",indces)

    df['Ambiguous1'] = Xtest[391:392:1,::1].reshape(54,).tolist()
    df['Ambiguous2'] = Xtest[430:431:1,::1].reshape(54,).tolist()
    df['Ambiguous3'] = Xtest[396:397:1,::1].reshape(54,).tolist()

    #plot the first ambigous predictions label agains the expected label
    fig4, ax = plt.subplots()
    ax.scatter(np.arange(len(df['readMe'])),df['Ambiguous1'],label ="Ambiguous Prediction1 Feature")
    ax.scatter(np.arange(len(df['readMe'])), df['lemda0'],label = "Expected Frequency Lemda0")
    ax.scatter(np.arange(len(df['readMe'])), df['lemda1'],label = "Expected Frequency Lemda1")
    ax.xaxis.set_ticks(np.arange(len(df['readMe'])))
    ax.xaxis.set_ticklabels(df['readMe'], rotation = 90)
    plt.xlabel("labels")
    plt.ylabel("frequency")
    plt.title("ambiguous prediction1 label compare vs expected")
    plt.legend()
    plt.show()

    #plot the first ambigous predictions label agains the expected label
    fig5, ax = plt.subplots()
    ax.scatter(np.arange(len(df['readMe'])),df['Ambiguous2'],label ="Ambiguous Prediction2 Feature")
    ax.scatter(np.arange(len(df['readMe'])), df['lemda0'],label = "Expected Frequency Lemda0")
    ax.scatter(np.arange(len(df['readMe'])), df['lemda1'],label = "Expected Frequency Lemda1")
    ax.xaxis.set_ticks(np.arange(len(df['readMe'])))
    ax.xaxis.set_ticklabels(df['readMe'], rotation = 90)
    plt.xlabel("labels")
    plt.ylabel("frequency")
    plt.title("ambiguous prediction2 label compare vs expected")
    plt.legend()
    plt.show()

    #plot the first ambigous predictions label agains the expected label
    fig6, ax = plt.subplots()
    ax.scatter(np.arange(len(df['readMe'])),df['Ambiguous3'],label ="Ambiguous Prediction3 Feature")
    ax.scatter(np.arange(len(df['readMe'])), df['lemda0'],label = "Expected Frequency Lemda0")
    ax.scatter(np.arange(len(df['readMe'])), df['lemda1'],label = "Expected Frequency Lemda1")
    ax.xaxis.set_ticks(np.arange(len(df['readMe'])))
    ax.xaxis.set_ticklabels(df['readMe'], rotation = 90)
    plt.xlabel("labels")
    plt.ylabel("frequency")
    plt.title("ambiguous prediction3 label compare vs expected")
    plt.legend()
    plt.show()

    #calculate the prediction probability for the top three most ambiguous
    ambig1PredP = math.exp(predX0[391])/ (math.exp(predX0[391]) + math.exp(predX1[391]))
    ambig2PredP = math.exp(predX0[430]) / (math.exp(predX0[430]) + math.exp(predX1[430]))
    ambig3PredP = math.exp(predX0[396]) / (math.exp(predX0[396]) + math.exp(predX1[396]))
    print("nonSpam predict probability", ambig1PredP,ambig2PredP,ambig3PredP)