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
import theano
import theano.tensor as T


import os
import sys
import math


class KNN():

    def __init__(self):
        pass

    #def get_euclidean_dist(self,x):
    #    return np.sum(((self.input.T - x).T ** 2),1)


    def get_nearest(self, input):
       
        x = T.dvector('x')
        #dists = self.get_euclidean_dist(x) 
        dists = T.sum(((input - x) ** 2),1)
        return theano.function([x], T.argmax(dists,0)) 

    def get_euclidean_dist(self, x,y):
        d = T.dot(x,y.T)
        d *= -2.0
        d += T.sum(x*x, axis=1).dimshuffle(0,'x')
        d += T.sum(y*y, axis=1)
        # Rounding errors occasionally cause negative entries in d
        d = d * T.cast(d>0,theano.config.floatX)
        return T.sqrt(d)

    def sort_dists(self, k, dists):
        sorted_indices = T.argsort(dists,axis=1)
        return sorted_indices[:,0:k]


    def error_test(self, pred_y, t):
        return T.mean(T.neq(pred_y, t))


    def get_k_neighbours_labels(self, train_set, test_set, num_cases,k):
        dists = self.get_euclidean_dist(test_set[0], train_set[0]) #find dist
        k_dists = T.sort(dists,1)[:,0:k] 
        k_argdists = T.argsort(dists,1)[:,0:k] 
        labels = train_set[1][k_argdists]
        return labels, k_dists, k_argdists

    def knearest_neighbours(self, train_set, test_set, num_cases,k): 
        labels, k_dists, k_argdists = self.get_k_neighbours_labels(train_set, test_set, num_cases,k)

        def pred(row, i, acc):
            bin_label = T.extra_ops.bincount(row)
            return T.set_subtensor(acc[i], T.argmax(bin_label))
    
        results, updates = theano.scan(fn=pred, outputs_info=T.zeros_like(test_set[1]),\
                             sequences=[labels, T.arange(num_cases)])
    
        pred_labels = results[-1] 
        return pred_labels, [k_dists, k_argdists]
    
    def classify(self, pred_labels, test_target ):
    
        err_class = T.mean(T.neq(pred_labels, test_target ))
        acc = (1-err_class)
        
        return acc



