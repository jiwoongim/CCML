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


import os, sys, math
import numpy as np

import theano 
import theano.tensor as T
import theano.tensor.slinalg as Tkron
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import pylab as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import *

TINY = np.exp(-19)

class ConvCNE():
    def __init__ (self, num_hid, num_dims, num_data, num_classes, batch_size, num_channels=1, nkerns=[10,10], poolsize=(2, 2), f0 = 5, f1 = 3):

        self.D =  int(np.sqrt(num_dims / num_channels))
        numpy_rng=np.random.RandomState(1234)
 
        self.image_shape0=[batch_size, num_channels, self.D, self.D]
        self.filter_shape0=(nkerns[0], num_channels, f0, f0) #TODO 
        img_out_size0= (self.D-f0+1)/poolsize[0]
        self.image_shape1=[batch_size, nkerns[0], img_out_size0, img_out_size0]
        self.filter_shape1=(nkerns[1], nkerns[0], f1, f1) #TODO 
        img_out_size1= (img_out_size0-f1+1)/poolsize[0]
        print 'Layer0: Filter size %d, image size out %d' % (f0, img_out_size0)
        print 'Layer1: Filter size %d, image size out %d' % (f1, img_out_size1)
        self.init_conv_filters(numpy_rng, self.D, nkerns, poolsize)

        convL1_dim = img_out_size1
        num_convH = nkerns[-1]*convL1_dim*convL1_dim
        A = 0.01 * np.asarray(numpy_rng.normal(size=(num_convH, num_hid)), dtype=theano.config.floatX)
        self.A = theano.shared(value=A, name='A', borrow=True)


        
        self.batch_size = batch_size
        self.num_hid    = num_hid
        self.num_cases   = num_data #X.get_value().shape[0] 
        self.num_classes = num_classes
        self.params = [self.A, self.W0, self.b0, self.W1, self.b1]


    def init_conv_filters(self,numpy_rng, D, nkerns,poolsize):

        ''' Convolutional Filters '''
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(self.filter_shape0[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (self.filter_shape0[0] * np.prod(self.filter_shape0[2:]) /
                   np.prod(poolsize))

        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W0 = theano.shared(
            np.asarray(
                numpy_rng.uniform(low=-W_bound, high=W_bound, size=self.filter_shape0),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        fan_in = np.prod(self.filter_shape1[1:])
        fan_out = (self.filter_shape1[0] * np.prod(self.filter_shape1[2:]) /
                   np.prod(poolsize))

        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W1 = theano.shared(
            np.asarray(
                numpy_rng.uniform(low=-W_bound, high=W_bound, size=self.filter_shape1),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values0 = np.zeros((self.filter_shape0[0],), dtype=theano.config.floatX)
        self.b0 = theano.shared(value=b_values0, borrow=True)
        b_values1 = np.zeros((self.filter_shape1[0],), dtype=theano.config.floatX)
        self.b1 = theano.shared(value=b_values1, borrow=True)


    def cost(self, X, y, k=5):

        num_cases = T.cast(X.shape[0], 'int32')
        p = self.compute_probability_matrix(X, y, num_cases, k=k)
        ##TODO compute new expectation
        e = T.sum(p[T.arange(num_cases, dtype='int32'), 0])
        f = -e/num_cases #- 0.1 * T.var(p[T.arange(num_cases), 0])

        return f


    def convProp(self,X, W, b, image_shape, filter_shape, poolsize=(2, 2)):

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=X,
            filters=W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        return T.tanh(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))


    def propagate(self, X, num_train=None):
        image_shape0 = self.image_shape0
        image_shape1 = self.image_shape1
        if num_train is not None:
            image_shape0[0] = num_train
            image_shape1[0] = num_train

        ConX = X.reshape(image_shape0)
        ConH0 = self.convProp(ConX , self.W0, self.b0, image_shape0, self.filter_shape0)
        ConH1 = self.convProp(ConH0, self.W1, self.b1, image_shape1, self.filter_shape1)
        H = ConH1.flatten(2)
        z = T.dot(H, self.A) #Transform x into z space 

        return z


    def compute_probability_matrix(self,X, y, num_cases, k=5):
        
        #X = dropout(self.rng, X, p=0.5)
        z = self.propagate(X) #Transform x into z space 
        dists = T.sqr(dist2hy(z,z))
        dists = T.extra_ops.fill_diagonal(dists, T.max(dists)+1)
        nv = T.min(dists,axis=1) # value of nearest neighbour 
        dists = (dists.T - nv).T

        #TODO change numerator
        d = T.extra_ops.fill_diagonal(dists, 0)
    
        #Take only k nearest 

        num = T.zeros((num_cases, 2))
        denom = T.zeros((num_cases,))

        #Mask for class i
        mask_i = T.eq(T.outer(T.ones_like(y),y).T, y)
        num,denom, nv = self.get_cond_prob(X,y, mask_i     , num, denom, d,num_cases, k, 0)  
        num,denom, nv = self.get_cond_prob(X,y, 1 - mask_i , num, denom, d,num_cases, k, 1, nv)  

        p = num / denom.dimshuffle(0,'x')    #prob that point i will be correctly classified    
        return p


    def get_cond_prob(self, X,y, mask_i, num, denom, d, num_cases, k, c_i, nv=None):
        #K nearest neighbour within a class i 
        dim_ci = T.sum(mask_i[0])
        k_indice = T.argsort(d*mask_i+(1-mask_i) * (T.max(d)+1),axis=1)[:,0:k]
       
        kd = T.zeros((num_cases,k))
        for it in xrange(k):
            kd = T.set_subtensor(kd[:,it], d[T.arange(num_cases, dtype='int32'),k_indice[:,it]]) 
        
        #Numerator
        if nv is None:
            nv = T.min(kd,axis=1) # value of nearest neighbour 
        kd = (kd.T - nv).T

        value = T.exp(-T.mean(kd,axis=1))
        num = T.set_subtensor(num[:,c_i], value) 

        denom += value  
    
        return num, denom, nv


    def weight_decay(self, wtype='l2'):
        if wtype=='l2':
            return 0.5*T.sum(self.A**2)
        elif wtype=='l1':
            return T.sum(abs(self.A))


