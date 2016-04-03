''' Version 1.000
 Code provided by Daniel Jiwoong Im and Graham W. Taylor 
 Permission is granted for anyone to copy, use, modify, or distribute this
 program and accompanying programs and documents for any purpose, provided
 this copyright notice is retained and prominently displayed, along with
 a note saying that the original programs are available from our
 web page.
 The programs and documents are distributed without any warranty, express or
 implied.  As the programs were written for research purposes only, they have
 not been tested to the degree that would be advisable in any important
 application.  All use of these programs is entirely at the user's own risk.'''

'''Demo of Learning a Metric for Class-Conditional KNN.
'''


import os, sys, math, cPickle, gzip, theano
import theano.tensor as T
import theano.tensor.slinalg as Tkron
import numpy as np

import pylab as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import *

TINY = np.exp(-19)

class NBKNCA():
    def __init__ (self, num_hid, num_dims, num_data, num_classes):

        numpy_rng=np.random.RandomState()
        A = 0.008 * np.asarray(numpy_rng.normal(size=(num_dims, num_hid)), dtype=theano.config.floatX)
        self.A = theano.shared(value=A, name='A', borrow=True)
        self.num_hid    = num_hid
        self.num_cases   = num_data 
        self.num_classes = num_classes
        self.params = [self.A]

    def cost(self, X, y, k=5):

        num_cases = X.shape[0]
        p = self.compute_probabilistic_matrix(X, y, num_cases, k=k)
        e = T.sum(p[T.arange(num_cases), y])
        f = -e/num_cases 

        return f

      
    def weight_decay(self):
        return 0.5*T.sum(self.A**2)

    def propagate(self, X, thrd=0.0):
        z = T.dot(X, self.A) #Transform x into z space 
        return z

    def compute_probabilistic_matrix(self,X, y, num_cases, k=5):

        z       = T.dot(X, self.A) #Transform x into z space 
        dists   = T.sqr(dist2hy(z,z))
        dists   = T.extra_ops.fill_diagonal(dists, T.max(dists)+1)
        nv      = T.min(dists,axis=1) # value of nearest neighbour 
        dists   = (dists.T - nv).T
        d       = T.extra_ops.fill_diagonal(dists, 0)
   
        #Take only k nearest 
        num     = T.zeros((num_cases, self.num_classes))
        denom   = T.zeros((num_cases,))
        for c_i in xrange(self.num_classes):

            #Mask for class i
            mask_i = T.eq(T.outer(T.ones_like(y),y),c_i)

            #K nearest neighbour within a class i 
            dim_ci = T.sum(mask_i[0])
            d_c_i = T.reshape(d[mask_i.nonzero()],(num_cases,dim_ci))
            k_indice = T.argsort(d_c_i, axis=1)[:,0:k]
            
            kd = T.zeros((num_cases,k))
            for it in xrange(k):
                kd = T.set_subtensor(kd[:,it], d_c_i[T.arange(num_cases),k_indice[:,it]]) 

            #Numerator
            value   = T.exp(-T.mean(kd,axis=1))
            num     = T.set_subtensor(num[:,c_i], value) 
            denom   += value 
            

        p = num / denom.dimshuffle(0,'x')    #prob that point i will be correctly classified    
        return p


    def weight_decay(self):
        return 0.5*T.sum(self.A**2)


