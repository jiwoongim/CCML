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
import pylab as pl
from sklearn.datasets import make_circles

from knn import *


def gaussian_log_prob(dists, variances):

    log_prob =  - (dists ** 2 / 2)# * variances.dimshuffle(0,'x')) \
            #- T.log(( T.sqrt(variances*2*np.pi))).dimshuffle(0,'x')
    return log_prob

def get_prior(num_cases_per_class, num_classes):
    
    num_cases_per_class = np.asarray(num_cases_per_class, dtype='float32')
    prior = num_cases_per_class / np.sum(num_cases_per_class) 
    return prior


def naive_bayes_nearest_neighbours(knn, train_set, test_set, num_classes,\
                                    num_cases_per_class, Nt, k):

    prior = get_prior(num_cases_per_class, num_classes)
    log_likelihood_classes = T.zeros((Nt, num_classes))
    dists_c = []
    examples_var = T.zeros((Nt,num_classes*k))
    dists_list = []
    for class_i in xrange(num_classes):
        num_cases = num_cases_per_class[class_i]
        dists = knn.get_euclidean_dist(test_set[0], train_set[class_i][0]) #find dist
        examples_var = T.set_subtensor(examples_var[:,class_i*k:(class_i+1)*k], dists[:,0:k])
        dists_list.append(dists)

    variances = T.var(examples_var, axis=1);#variances / num_classes


    for class_i in xrange(num_classes):
        num_cases = num_cases_per_class[class_i]
        dists = dists_list[class_i]
        k_dists = T.sort(dists,1)[:,0:k] 


        log_probs_class_i = gaussian_log_prob(k_dists, variances)
        log_probs_class_i = T.sum(log_probs_class_i,axis=1) -np.log(k) #/ k
        log_likelihood_classes = T.set_subtensor(log_likelihood_classes[:,class_i], log_probs_class_i)


    pred_c = T.argmax(log_likelihood_classes, axis=1)
    acc = knn.classify(pred_c, test_set[1]).eval()

    return acc


