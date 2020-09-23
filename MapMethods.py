# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:31:14 2020

@author: orica
"""
import numpy as np
import scipy as sp
from numpy import linalg as la
import numpy.matlib
from matplotlib import pyplot as plt
#import globalVals

def FrobNorm(A,B):
    return np.trace(np.transpose(B)@A)

############################### TSCA ########################################
class TSCA:
# Implementation of TSCA
    def Func(Z,X,Y,gamma = (1,-0.05),reduceComp = 1):
        C = []
        if X.shape[0] == X.size: # if it is a vector
            T = X.size
            C.append(np.outer(X,X)/T) # estimate autocorrelation matrix
        else: # it is already a square autocorrelation matrix
                C.append(X)
                T = X.shape[0]
        for i in Y: # append y's to C (if it is a vector calc autocorr mat)
            C.append(np.outer(i,i)/T) if i.shape[0] == i.size else C.append(i)
        N = len(C)
        MAT = np.zeros((N,N))
        Vec = np.zeros(N)
        for i in range(N):
            for j in range(N):
                MAT[i,j] = FrobNorm(C[j],C[i])
            Vec[i] = gamma[0]*np.trace(C[i]) if i==0 else gamma[1]*np.trace(C[i])
        Alpha = la.inv(MAT)@Vec
        Q = np.zeros((T,T))
        for i in range(N):
            Q = Q + Alpha[i]*C[i]    
        if reduceComp==0:
           M = Z@Q@np.transpose(Z)
           D,components = la.eig(M)
           ind = np.argsort(D)
           ind = ind[::-1]
           D = D[ind]
           components = components[:,ind]
        else:
           U,tempS,V = la.svd(Z,full_matrices=False)
           S = np.zeros((T,T))
           np.fill_diagonal(S,tempS)
           M = S@np.transpose(V)@Q@V@np.transpose(S)
           D,W = la.eig(M)
           D = np.absolute(D)
           ind = np.argsort(D) 
           ind = ind[::-1]
           D = D[ind]
           W = W[:,ind]
           components = U@W
        components = components*np.matlib.repmat(np.sign(np.mean(components,axis=0)),components.shape[0],1)
        projected = np.transpose(components)@Z
        if X.shape[0] == X.size: # if it is a vector, fix inversion
            components[:,0] = components[:,0]*np.sign(np.dot(projected[0,:],X)) 
          
        output = {
               "projected" : projected,
               "components" : components,
               "D" : D,
               "Alpha" : Alpha,
               "gammas" : gamma,
               "C" : C
               }
        return output
    # Analysis to view the components, eigenvalues and time coarse projections
    def Analyze(struct,numComp = 5,Title = [],largeComp = 1,T = 800):
        components = struct["components"]
        projected = struct["projected"]
        fig1  = plt.figure()
        fig1.suptitle('Spatial Components')
        fig2  = plt.figure()
        fig2.suptitle('Time Course')
        for i in range(numComp*numComp):
            ax = fig1.add_subplot(numComp, numComp, i+1)
            ax.imshow(np.absolute(components[:,i].reshape((globalVals.imgSize[0],globalVals.imgSize[1]))))
            ax.set_title('component %d' %(i+1))
            
            ax = fig2.add_subplot(numComp, numComp, i+1)
            ax.plot(np.absolute(projected[i,:]))
            ax.set_title('component %d' %(i+1))
            
        plt.figure()
        plt.plot(struct['D'],'bo')
        plt.title('eigen-values in descending order')
        return
################################ END TSCA ###################################
#===========================================================================#    
################################ GLM ########################################
def GLM(Z,noiseFreqs,responseSigs):
    fs = 100
    T = Z.shape[1] # total time
    Z = Z.T
    t = np.linspace(0,(T-1)/fs,T) # time vector
    X1 = np.ones((T,1)) # DC
    Xn = np.zeros((T,len(noiseFreqs)*2))
    k=0
    responseSigs = np.asarray(responseSigs).T
    for i in noiseFreqs:
        Xn[:,k] = np.sin(2*np.pi*i*t) 
        Xn[:,k+1] = np.cos(2*np.pi*i*t)
        k+=2
    Xtot = np.concatenate((X1,Xn,responseSigs),1)
    beta = la.inv(Xtot.T@Xtot)@Xtot.T@Z
    All = Xtot@beta
    Residuals = Z - All
    Signal = responseSigs@beta[-responseSigs.shape[1]:,:]+Residuals
    return beta[-responseSigs.shape[1]:,:],Signal
################################ END GLM ####################################
#===========================================================================#    
################################ TMAX #######################################
def Tmax(Z,responseSig,r_thresh,gaussSTD = 0):
    return
################################ END TMAX ###################################
#===========================================================================#    
################################ CORR #######################################
def Corr(Z,responseSig):
responseSig = loadmat(r'responseSig.mat')
responseSig = np.concatenate((responseSig["responseSig"][0],np.zeros(700)))
theoreticalSigs = [np.roll(responseSig,100*i) for i in range(8)]
corrArray = np.zeros((nrow,ncol,8))
responseSources = np.zeros((nrow, ncol))
responseSourcesIndexes = np.zeros((nrow, ncol))
for i in range(nrow):
    for j in range(ncol):
        for m in range(8):
         corrArray[i,j,m] = np.corrcoef(Z[i,j,100*m:100*(m+1)], theoreticalSigs[m][100*m:100*(m+1)])[1,0]
for i in range(nrow):
    for j in range(ncol):
          responseSources[i,j] = np.max(corrArray[i,j,:])
          responseSourcesIndexes[i,j] = np.argmax(corrArray[i,j,:])
prctile = np.percentile(responseSources,95)
responseSourcesIndexes[responseSources<prctile] = 0
plt.imshow(responseSourcesIndexes) 
     return
################################ END CORR ###################################
#===========================================================================# 
################################ AOF #######################################
def AOF(Z,N=8):
    if len(Z.shape)!=3:
        print('please reshape Z to be 3d')
        return
    AOF = []
    for i in range(N):
        ax = fig2.add_subplot(1, 8, i+1)
        temp = np.mean(Z[:,:,i*100+10:i*100+15],2)
        temp = (temp-np.min(temp))/(np.max(temp)-np.min(temp))
        AOF.append(temp)
        #plt.imshow(temp) 
    return AOF
################################ END AOF ###################################
#===========================================================================#

