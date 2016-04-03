''' Version 1.000
 
 Code provided by Jiwoong Im

 Permission is granted for anyone to copy, use, modify, or distribute this
 program and accompanying programs and documents for any purpose, provided
 this copyright notice is retained and prominently displayed, along with
 a note saying that the original programs are available from our
 web page.
 The programs and documents are distributed without any warranty, express or
 implied.  As the programs were written for research purposes only, they have
 not been tested to the degree that would be advisable in any important
 application.  All use of these programs is entirely at the user's own risk.'''


import numpy as np
import scipy as sp
import pylab as pl
import scipy.spatial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import sys
import math

#################
#global parameter
#################
dimR =2; #number of princial components
vis1 = 1; #vis is set to 1 to view the graphs
scale1 = 1;
scale2 = 1;

#Computing eigenvalues and eigenvectors
def compute_eig(x, dimR):
    #xMean = np.mean(x, axis=0);
    xMean = x.mean(0)
    tmp = x- xMean;
    covM = np.cov(tmp.T)
    [eVal, eVec] = np.linalg.eig(covM);

    order = np.argsort(eVal, axis=0);
    order = order[::-1];
    
    #import pdb; pdb.set_trace()
    eVal = eVal[order];
    base = eVec[:,order];
    #base = base[:,0:dimR];
    
    return [base,eVal, order];

#project data
def project(x, dimR, base, X=None):

    D,N = x.shape
    # If the total data set (train+test), set total data set to be train. This is not really important
    if X is None: 
        X = x

    projX = np.dot(X, base[:,0:dimR]);
    return projX

# evaluate
''' base - pricipal bases / matrix '''
def evaluate(x, xt, dimR, base, X, ax2):

    #projecting train data
    projX = project(x, dimR, base, X);

    #projecting test data
    projXt = project(xt, dimR, base, X);
   
    #1-distanced
    Y = scipy.spatial.distance.cdist(projX.T, projXt.T, 'euclidean');
    xt1 = xt[:, np.argmin(Y, axis=0)<= x.shape[1]/2];
    xt2 = xt[:, np.argmin(Y, axis=0)>  x.shape[1]/2];
    projXt1 = projXt[:, np.argmin(Y, axis=0)<= x.shape[1]/2];
    projXt2 = projXt[:, np.argmin(Y, axis=0)> x.shape[1]/2];

    #visualize projected Data
    ax2.plot(projX[0,:x.shape[1]/2], projX[1,:x.shape[1]/2], projX[2,:x.shape[1]/2], 'o', label='class1 train data projected ');
    ax2.plot(projX[0,x.shape[1]/2:], projX[1,x.shape[1]/2:], projX[2,x.shape[1]/2:],'o', label='class2 train data projected ');
    ax2.plot(projXt1[0,:], projXt1[1,:], projXt1[2,:], 'o', label='class1 test data projected ');
    ax2.plot(projXt2[0,:], projXt2[1,:], projXt2[2,:], 'o', label='class2 test data projected ');

    return [xt1, xt2];

