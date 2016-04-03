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

import os, sys, math, timeit
import cPickle as pickle
import numpy as np
import pylab as pl
import scipy.io as spio
import scipy.io 


from knn import *
from utils import *
from nbknca import *
from optimize import *
import conv_cne as cccml2
from naive_bayes_nearest_neighbours import *


#current_dir  = '/export/mlrg/imj/machine_learning/research/distance_metric/codes/params/'
current_dir  = '/groups/branson/home/imd/Documents/machine_learning_uofg/research/distance_metric/codes/params/'

def train_model(train_set, valid_set, num_dims, num_cases, num_hid, num_classes,\
                                 num_epoch=100, batch_sz=5000, epsilon = 0.01, \
                                momentum=0., weightcost=0.002, k=5, trainF=True, model_type='cccml'):

    hyper_params = [batch_sz, epsilon, momentum, weightcost] 
    if model_type=='cccml':
        model = cccml.ConvCNE(num_hid, num_dims, num_cases, num_classes, batch_sz, num_channels=2)
        param_fname = 'Optimal_Conv_NORB'
    else:
        model = cccml2.ConvCNE(num_hid, num_dims, num_cases, num_classes, batch_sz, num_channels=2,f1=3,f0=3)
        param_fname = 'Optimal_Conv2_NORB'

    gmd = GraddescentMinibatch(hyper_params)
    train_update, get_valid_cost = gmd.stochastic_update(model, train_set, valid_set)
    num_batches = int(num_cases / batch_sz)

    Nv = valid_set[1].eval().shape[0]
    num_valid_batches = int(Nv / batch_sz)
    max_valid_exp = 0
    best_epoch = num_epoch+1
    for epoch in xrange(num_epoch+1):

        average_expectation = []
        start = timeit.default_timer()
        for batch_i in xrange(num_batches):

            eps = get_epsilon(epsilon, 100, epoch)
            mom = get_epsilon(momentum, 300, epoch)
            perm_ind = np.random.permutation(num_cases).flatten()[0:batch_sz]    
            mini_expectation = - train_update(perm_ind, eps, mom) * batch_sz
            average_expectation.append(mini_expectation)

        stop = timeit.default_timer()
        if epoch % 5 == 0:
            expectation_train = np.ma.average(np.asarray((average_expectation)))
            print '...Epoch %d, Learning Rate: %g, Expectation %g, Time %g' \
                            % (epoch, eps, expectation_train, stop - start)

        
        if epoch % 10 == 0 and epoch > num_epoch * 0.45 and trainF:

            valid_expectation = []
            for k in xrange(num_valid_batches):
                valid_expectation_k = - get_valid_cost(k) * batch_sz 
                        #* valid_set[0][k*batch_sz:(k+1)*batch_sz,:].get_value(borrow=True).shape[0] 
                valid_expectation.append(valid_expectation_k)
            valid_expectation = np.mean(np.asarray(valid_expectation))

            print '...Expectation on Validation Data Set %f' % (valid_expectation)   

            if max_valid_exp <= valid_expectation:
                max_valid_exp = valid_expectation
                save_the_weight(model.params, current_dir+param_fname) 
                best_epoch = epoch

    if model_type=='cccml':
        path1 = current_dir+param_fname+'.save'
        model.A, model.W, model.b = unpickle(path1)
    elif model_type=='cccml2':
        path1 = current_dir+param_fname+'.save'
        model.A, model.W0, model.b0, model.W1, model.b1 = unpickle(path1)

    valid_expectation = []
    for k in xrange(num_valid_batches):
        valid_expectation_k = - get_valid_cost(k) * batch_sz 
                #* valid_set[0][k*batch_sz:(k+1)*batch_sz,:].get_value(borrow=True).shape[0] 
        valid_expectation.append(valid_expectation_k)
    valid_expectation = np.mean(np.asarray(valid_expectation))
    print '-->Cost on valid Data Set %g, Best Epoch at %d' % (valid_expectation, best_epoch)   
    print "-->Settings: lr %g, wc %g, n_hid %s, n_dim %d, n_epo %d" % \
                (epsilon, weightcost, str(num_hid), num_dims, num_epoch)
    print '-->Parameter is saved at %s' % param_fname
    return model, best_epoch


def run(train_set, valid_set, test_set, knn, hyper_params):

    num_dims, num_train_cases,num_test_cases, num_hid, num_classes, num_epoch, epsilon, batch_sz,\
                                        k, model_type, weightcost  = hyper_params
    knn_acc=0; cknn_acc=0; acc3=0; acc4=0
    #Training model
    print '... Training New Model'
    model, best_epoch = train_model(train_set, valid_set, num_dims, num_train_cases, num_hid, num_classes,\
                weightcost=weightcost, num_epoch=num_epoch, batch_sz=batch_sz, \
                epsilon=epsilon,k=k,model_type=model_type)

    X = np.concatenate([train_data, valid_data], axis=0)
    Y = np.concatenate([train_labels, valid_labels], axis=0)
    train_set = shared_dataset([X, Y]);    

    ## Applying K-NearestNeighbours in the data space
    print '... Computing KNN'
    pred_labels, k_dists = knn.knearest_neighbours(train_set, test_set, num_test_cases, k) 
    knn_acc = knn.classify(pred_labels,test_set[1]).eval()

    ##Applying Naive Bayesian K-Nearest Neighbours 
    print '... Computing Conditioanl-KNN'
    train_sets_per_class, num_cases_per_class = \
            separate_data_into_classes([X, Y], num_classes)
    cknn_acc = naive_bayes_nearest_neighbours(knn, train_sets_per_class, test_set, num_classes, \
                                    num_cases_per_class, num_test_cases, k)

    ## Applying K-NearestNeighbours in better distance metric space
    Ntot = train_set[0].eval().shape[0]
    H_train = [model.propagate(train_set[0], Ntot), train_set[1]] 
    H_test  = [model.propagate(test_set[0], num_test_cases), test_set[1]]   

    pred_labels, k_dists = knn.knearest_neighbours(H_train, H_test, num_test_cases, k)
    acc3 = knn.classify(pred_labels,test_set[1]).eval()

    ##Applying Naive Bayesian K-Nearest Neighbours in the better distance metric  
    train_sets_per_class, num_cases_per_class = \
                        separate_data_into_classes([H_train[0].eval(), Y], num_classes)
    acc4 = naive_bayes_nearest_neighbours(knn, train_sets_per_class, H_test, num_classes, \
                                   num_cases_per_class, num_test_cases, k)

    ##print knn_acc, cknn_acc
    print knn_acc, cknn_acc, acc3, acc4
    return knn_acc, cknn_acc, acc3, acc4


#HYPER-PARAMETERS
k           = 13
DIMR        = 850
PCA_flag    = False
LCN_flag    = False
num_epoch   = 350
#model_type  = 'nbknca'
model_type  = 'cccml2'
trainF      = True

if model_type == 'cccml':
    weightcost  = 0.0006; epsilon = 0.06; batch_sz = 150; k=13; num_hid=600
elif model_type == 'cccml2':
    weightcost  = 0.0015; epsilon = 0.07; batch_sz = 250; k=13; num_hid=400; num_epoch = 500
    #weightcost  = 0.0006; epsilon = 0.006; num_hid = [500,300]; 


if __name__ == '__main__':

    ## Load SMALL norb Sample Data
    norb_path = '/groups/branson/home/imd/Documents/machine_learning_uofg/data/NORB/'  
    save = []

    knn = KNN()

    #for DIMR in [100,300,500, 600, 800, 1000]:#, 850, 1024]:
    #    num_hid = DIMR
    #for num_hid in [350,400, 450,500]:
    #for DIMR in [50,70, 80, 100, 120, 140, 160, 180, 200]:
    #    num_hid = DIMR
    #for num_hid in [[400,400], [400,500], [400,600]]:
    #for batch_sz in [100, 150, 250, 350]:#, 450, 550, 650, 700, 750, 1000, 1500, 2000, 3000, 4000, 5000]:
    #for epsilon in [0.07, 0.06, 0.05,0.04,0.03]:
    #for k in [3,5,7,9,11, 13,15]:
    #for k in [9,10,11,12,13,14]:
    #for weightcost in [0.0008, 0.0006, 0.0004,0.0001, 0]:
    #for weightcost in [0.001, 0.0013, 0.0015, 0.0017,0.002]:
    #for num_epoch in [400, 500, 550]:
    for tmp in [1]:
        train_data, test_data, train_labels, test_labels, train_info, test_info \
                =  load_cached_data(filename=norb_path+'smallnorb.pkl.gz')

        train_data, train_labels, valid_data, valid_labels, test_data, base = \
                        preprocess(train_data, train_labels, test_data, PCA=PCA_flag, DIMR=DIMR)

        num_train_cases, num_dims = train_data.shape
        num_valid_cases, num_dims = valid_data.shape
        num_test_cases , num_dims = test_data.shape
        num_classes = int(np.max(train_labels)+1)


        print '# of Training exmaples: %d\n# of Validation exmaples: %d\n# of Dimensionality %d' \
                                    % (num_train_cases, num_valid_cases, num_dims)

       
        train_set = shared_dataset([train_data, train_labels]);    
        test_set  = shared_dataset([test_data,  test_labels ]);    
        valid_set = shared_dataset([valid_data, valid_labels]); 

        hyper_params=[num_dims, num_train_cases,num_test_cases, num_hid, num_classes, \
                        num_epoch, epsilon, batch_sz, k, model_type, weightcost]
        knn_acc, cknn_acc, acc3, acc4 = run(train_set, valid_set, test_set, knn, hyper_params)
        save.append([knn_acc, cknn_acc, acc3, acc4])
 
    print save


