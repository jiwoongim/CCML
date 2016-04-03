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

import numpy as np
import pylab as pl
import timeit
import os, sys, math
import scipy.io 

from knn import *
from utils import *
from nbknca import *
from optimize import *
from conv_cne import *
from naive_bayes_nearest_neighbours import *

#TODO NEED TO CHANGE THIS PATH
current_dir  = '/groups/branson/home/imd/Documents/machine_learning_uofg/research/distance_metric/codes/params/'

def train_model(train_set, valid_set, num_dims, num_cases, num_hid, num_classes,\
                                 num_epoch=100, batch_sz=5000, epsilon = 0.01, \
                    momentum=0.0, weightcost=0.002, k=5, model_type='cne', trainF=True):

    hyper_params = [batch_sz, epsilon, momentum, weightcost] 

    if model_type == 'nbknca' : 
        model = NBKNCA(num_hid, num_dims, num_cases, num_classes)
    elif model_type == 'cne' : 
        model = ConvCNE(num_hid, num_dims, num_cases, num_classes, batch_sz)

    gmd = GraddescentMinibatch(hyper_params)
    train_update, get_valid_cost = gmd.batch_update(model, train_set, valid_set)
    get_grad = gmd.compute_grad(model, train_set)

    num_batches = int(num_cases / batch_sz)

    Nv = valid_set[1].eval().shape[0]
    num_valid_batches = int(Nv / batch_sz)

    max_valid_exp = 0
    param_fname = 'test_conv2_mnist'
    best_epoch = num_epoch+1
    for epoch in xrange(num_epoch+1):

        average_expectation = []
        start = timeit.default_timer()
        for batch_i in xrange(num_batches):

            eps = get_epsilon(epsilon, 100, epoch)
            mom = get_epsilon(momentum, 300, epoch)


            mini_expectation = - train_update(batch_i, eps, mom) * batch_sz
            average_expectation.append(mini_expectation)
        stop = timeit.default_timer()
        if epoch % 5 == 0:
            expectation_train = np.ma.average(np.asarray((average_expectation)))
            print '...Epoch %d, Learning Rate: %g, Expectation %g, Time %g' \
                            % (epoch, eps, expectation_train, stop - start)

        
        if epoch % 5 == 0 and epoch > num_epoch * 0.5:
            valid_expectation = []
            for k in xrange(num_valid_batches):
                valid_expectation_k = - get_valid_cost(k) * batch_sz 
                valid_expectation.append(valid_expectation_k)
            valid_expectation = np.mean(np.asarray(valid_expectation))
            print '...Expectation on Validation Data Set %f' % (valid_expectation)   

            if max_valid_exp <= valid_expectation and trainF:
                max_valid_exp = valid_expectation
                save_the_weight([model.A, model.W0, model.b0, model.W1, model.b1],\
                                                        current_dir+param_fname) 
                best_epoch = epoch

    if trainF:
        path1 = current_dir+param_fname+'.save'
        model.A, model.W0, model.b0, model.W1, model.b1 = ld.unpickle(path1)

    valid_expectation = []
    for k in xrange(num_valid_batches):
        valid_expectation_k = - get_valid_cost(k) * valid_set[0].get_value(borrow=True).shape[0] 
        valid_expectation.append(valid_expectation_k)
    valid_expectation = np.mean(np.asarray(valid_expectation))
    print 'Cost on valid Data Set %g, Best Epoch at %d' % (valid_expectation, best_epoch)   
    print "Settings: lr %g, wc %g, n_hid %d, n_dim %d, n_epo %d" % \
                (epsilon, weightcost, num_hid, num_dims, num_epoch)
    print '-->Parameter is saved at %s' % param_fname
    return model, best_epoch

def run(train_set, valid_set, test_set, knn, hyper_params):

    num_dims, num_train_cases, num_test_cases, num_hid, num_classes, num_epoch, epsilon, batch_sz, \
                            k, model_type, weightcost = hyper_params

    #Training model
    print '... Training New Model'
    model, best_epoch = train_model(train_set, valid_set, num_dims, num_train_cases, num_hid, num_classes,\
                    weightcost=weightcost, num_epoch=num_epoch, batch_sz=batch_sz,\
                    epsilon=epsilon,k=k, model_type=model_type)

    X = np.concatenate([train_set_t[0], valid_set_t[0]], axis=0)
    Y = np.concatenate([train_set_t[1], valid_set_t[1]], axis=0)
    train_set = ld.shared_dataset([X, Y]);    

    ## Applying K-NearestNeighbours in the data space
    print '... Computing KNN'
    pred_labels, k_dists = knn.knearest_neighbours(train_set, test_set, num_test_cases, k) 
    knn_acc = knn.classify(pred_labels,test_set[1]).eval()

    ##Applying Naive Bayesian K-Nearest Neighbours 
    print '... Computing Conditioanl-KNN'
    train_sets_per_class, num_cases_per_class = \
            separate_data_into_classes([X, Y], num_classes)
    ccknn_acc = naive_bayes_nearest_neighbours(knn, train_sets_per_class, test_set, num_classes, \
                                    num_cases_per_class, num_test_cases, k)

    ## Applying K-NearestNeighbours in better distance metric space
    NTot= train_set[0].eval().shape[0]
    convH_tr = model.propagate(train_set[0], NTot)#(NTot,1,model.D,model.D))
    convH_te = model.propagate(test_set[0] , num_test_cases) #(num_test_cases ,1,model.D,model.D))

    H_train = [convH_tr, train_set[1]] 
    H_test  = [convH_te, test_set[1]]   
    pred_labels, k_dists = knn.knearest_neighbours(H_train, H_test, num_test_cases, k)
    acc3 = knn.classify(pred_labels,test_set[1]).eval()
     
    ##Applying Naive Bayesian K-Nearest Neighbours in the better distance metric  
    train_sets_per_class, num_cases_per_class = \
                        separate_data_into_classes([H_train[0].eval(), Y], num_classes)
    acc4 = naive_bayes_nearest_neighbours(knn, train_sets_per_class, H_test, num_classes, \
                                   num_cases_per_class, num_test_cases, k)

    ##print knn_acc, cknn_acc
    #print knn_acc, ccknn_acc, acc3, acc4
    return knn_acc, ccknn_acc, acc3, acc4


#HYPER-PARAMETERS
k           = 7
DIMR        = 250
PCA_flag    = False
LCN_flag    = False
num_epoch   = 270
num_hid     = DIMR
batch_sz    = 1000
#model_type  = 'nbknca'
model_type  = 'cne'
if model_type == 'nbknca':
    weightcost  = 0.005; epsilon     = 0.1
elif model_type == 'cne':
    weightcost  = 0.001; epsilon     = 0.135#0.115

if __name__ == '__main__':

    ## Load MNIST Sample Data
    #mnist_path = '/export/mlrg/imj/machine_learning/data/MNIST/mnist.pkl.gz'
    mnist_path = '/groups/branson/home/imd/Documents/machine_learning_uofg/data/MNIST/mnist.pkl.gz'

    #lmnn_results = scipy.io.loadmat('lmnn_mnist9.mat')
    knn = KNN()
    save = []

    #for num_hid in [50,100,150,200,270,350,400,500]:
    #num_hid = DIMR
    #for k in [3,4,5,7]:
    #    print k
    #for DIMR in [150,350, 400, 450]:
    #for batch_sz in [2000, 3000, 4000, 5000,6000,7000]:
    #for batch_sz in [200,500,1000]:
    #for weightcost in [0.001, 0.0008, 0.0006]:
    for epsilon in [0.1, 0.075]:
    #for tmp in [1]:

        train_set_t, valid_set_t, test_set_t = load_dataset(mnist_path) 
        train_set_t[0], valid_set_t[0],test_set_t[0], base = preprocess(train_set_t[0],\
                    train_set_t[1],test_set_t[0], valid_set_t[0], PCA=PCA_flag, DIMR=DIMR)

        num_train_cases, num_dims = train_set_t[0].shape
        num_valid_cases, num_dims = valid_set_t[0].shape
        num_test_cases , num_dims = test_set_t[0].shape
        num_classes = int(np.max(train_set_t[1])+1)

        print '# of Training exmaples: %d\n# of Validation exmaples: %d\n# of Dimensionality %d' \
                                    % (num_train_cases, num_valid_cases, num_dims)

        train_set = shared_dataset(train_set_t);    
        test_set  = shared_dataset( test_set_t);    
        valid_set = shared_dataset(valid_set_t); 

        knn = KNN()
        hyper_params=[num_dims, num_train_cases, num_test_cases, num_hid, num_classes, \
                        num_epoch, epsilon, batch_sz, k, model_type, weightcost]
        knn_acc, cknn_acc, acc3, acc4 = run(train_set, valid_set, test_set, knn, hyper_params)
        save.append([knn_acc, cknn_acc, acc3, acc4])
   
    print save



