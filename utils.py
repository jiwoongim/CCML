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
import numpy as np
import pylab as pl
from joblib import Parallel, delayed
from scipy.sparse import hstack

from pca import *


'''decaying learning rate'''
def get_epsilon(epsilon, n, i):
    return epsilon / ( 1 + i/float(n))


def shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """

    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    #When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue

    return shared_x, T.cast(shared_y, 'int32')


def load_dataset(path):
    # Load the dataset
    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    return [train_set[0], train_set[1]], \
            [valid_set[0],valid_set[1]], \
            [test_set [0],test_set [1]]


def load_cached_data(filename='/tmp/smallnorb.pkl.gz'):
    """ Load data from cache """

    with gzip.open(filename, 'r') as file:
        train_data, test_data, train_labels, test_labels, train_info, test_info = \
                    cPickle.load(file)

    return train_data, test_data, train_labels, test_labels, train_info, test_info



def separate_data_into_classes(train_set, num_classes):
   
    sep_train_set = []
    num_cases_per_class = []

    for class_i in xrange(num_classes):
        train_data = train_set[0][train_set[1]==class_i,:]
        Nc = train_data.shape[0]
        num_cases_per_class.append(Nc)
        sep_train_set.append(shared_dataset([train_data, class_i*np.ones((Nc,1),dtype='float32')]))
    return sep_train_set, num_cases_per_class


def normalize(data, vdata=None, tdata=None):
    mu   = np.mean(data, axis=0)
    std  = np.std(data, axis=0)
    data = ( data - mu ) / std

    if vdata == None and tdata != None:
        tdata = (tdata - mu ) /std
        return data, tdata

    if vdata != None and tdata != None:
        vdata = (vdata - mu ) /std
        tdata = (tdata - mu ) /std
        return data, vdata, tdata
    return data



def apply_pca(Xtrain,Xtest,Xvalid=None, dimR=50,base=None):

    print '...Applying PCA with %d principal component' % dimR

    if Xvalid is None:
        X = Xtrain
    else:
        X = np.concatenate([Xtrain, Xvalid], axis=0)
        #X = hstack((Xtrain.T, Xvalid.T))

    if base is None:
        [base, eVal, order] = compute_eig(X, dimR); #computing eigenvalues, eigenvectors
    
    #projecting train data
    projX = project(Xtrain, dimR, base);
    projX = projX.astype('float32')

    #projecting test data
    projXt = project(Xtest, dimR, base);
    projXt= projXt.astype('float32')

    #projecting Valid data
    if Xvalid is not None:
        projXv = project(Xvalid, dimR, base);
        projXv= projXv.astype('float32')

    #TODO check the reconstruction error!
    reconX = np.dot(projX, base[:,0:dimR].T);
    reconXt = np.dot(projXt, base[:,0:dimR].T);
    recon_err_X  = np.mean(np.sum((Xtrain - reconX)**2, axis=1))
    recon_err_Xt = np.mean(np.sum((Xtest - reconXt)**2, axis=1))

    residual_ratio_X = recon_err_X  / np.mean(np.sum(Xtrain**2,axis=1))
    residual_ratio_Xt= recon_err_Xt / np.mean(np.sum(Xtest **2,axis=1))
    print 'Residual ratio for the train %f, residual ratio for the test %f' % \
                                    (residual_ratio_X,residual_ratio_Xt)

    if Xvalid is not None:
        return projX, projXt, projXv, base  
    else:
        return projX, projXt, base

DIMR=150
def preprocess(Xtrain, ytrain, Xtest, Xvalid=None,\
                    normalize=False, PCA=False,LCN=False, DIMR=20, valid_ratio=0.2):

    base = []
    if LCN:
        sigmas = [1.591, 1.591]
        D = int(np.sqrt(Xtrain.shape[1]))
        Xtrain = np.asarray(Parallel(n_jobs=-1, verbose=10)(delayed(im_lcn_vec)(i, D, D,
                                                                        sigmas) \
                                               for i in Xtrain))
        Xtest = np.asarray(Parallel(n_jobs=-1, verbose=10)(delayed(im_lcn_vec)(i, D, D,
                                                                        sigmas) \
                                               for i in Xtest))
        Xvalid = np.asarray(Parallel(n_jobs=-1, verbose=10)(delayed(im_lcn_vec)(i, D, D,
                                                                        sigmas) \
                                               for i in Xvalid))

    if PCA:
        if Xvalid is None:
            Xtrain, Xtest, base = apply_pca(Xtrain, Xtest, Xvalid, dimR=DIMR)
        else:
            Xtrain, Xtest, Xvalid, base = apply_pca(Xtrain, Xtest, Xvalid, dimR=DIMR)

    if normalize: 
        if Xvalid is None:
            Xtrain, Xtest = normalize(Xtrain, tdata=Xtest)
        else:
            Xtrain, Xvalid, Xtest = normalize(Xtrain, vdata=Xvalid, tdata=Xtest)

    #Generate Validation dataset
    if Xvalid is None:
        N = Xtrain.shape[0]
        Nv= int(N * valid_ratio)
        Xvalid = Xtrain[0:Nv]
        yvalid = ytrain[0:Nv]
        Xtrain = Xtrain[Nv+1:]
        ytrain = ytrain[Nv+1:]

        return Xtrain, ytrain, Xvalid, yvalid, Xtest, base   

    return Xtrain, Xvalid, Xtest, base


def save_dataset(x,y,z,fname):
    f = file(fname+'.save', 'wb')
    cPickle.dump([x,y,z], f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def unpickle(path):
    ''' For cifar-10 data, it will return dictionary'''
    #Load the cifar 10
    f = open(path, 'rb')
    data = cPickle.load(f)
    f.close()
    return data 



def lcn_2d(im, sigmas=[1.591, 1.591]):
    ''' Apply local contrast normalization to a square image.
    Uses a scheme described in Pinto et al (2008)
    Based on matlab code by Koray Kavukcuoglu
    http://cs.nyu.edu/~koray/publis/code/randomc101.tar.gz

    data is 2-d
    sigmas is a 2-d vector of standard devs (to define local smoothing kernel)

    Example
    =======
    im_p = lcn_2d(im,[1.591, 1.591])
    '''

    #assert(issubclass(im.dtype.type, np.floating))
    im = np.cast[np.float](im)

    # 1. subtract the mean and divide by std dev
    mn = np.mean(im)
    sd = np.std(im, ddof=1)

    im -= mn
    im /= sd

    # # 2. compute local mean and std
    lmn = gaussian_filter(im, sigmas, mode='reflect')
    lmnsq = gaussian_filter(im ** 2, sigmas, mode='reflect')

    lvar = lmnsq - lmn ** 2;
    #lvar = np.where( lvar < 0, lvar, 0)
    np.clip(lvar, 0, np.inf, lvar)  # items < 0 set to 0
    lstd = np.sqrt(lvar)

    np.clip(lstd, 1, np.inf, lstd)

    im -= lmn
    im /= lstd

    return im




def lcn_cifar10(train_set_t, valid_set_t, test_set_t):
        
    D = int(np.sqrt(test_set_t[0].shape[1] / 3))
    sigmas = [1.591, 1.591]
    Xtrain1 = np.asarray(Parallel(n_jobs=-1, verbose=10)(delayed(im_lcn_vec)(i, D, D,
                                                                        sigmas) \
                                                for i in train_set_t[0][:,:1024]))
    Xtrain2 = np.asarray(Parallel(n_jobs=-1, verbose=10)(delayed(im_lcn_vec)(i, D, D,
                                                                        sigmas) \
                                                for i in train_set_t[0][:,1024:2048]))
    Xtrain3 = np.asarray(Parallel(n_jobs=-1, verbose=10)(delayed(im_lcn_vec)(i, D, D,
                                                                        sigmas) \
                                                for i in train_set_t[0][:,2048:3072]))   
    
    Xtest1 = np.asarray(Parallel(n_jobs=-1, verbose=10)(delayed(im_lcn_vec)(i, D, D,
                                                                   sigmas) \
                                           for i in test_set_t[0][:,:1024]))
    Xtest2 = np.asarray(Parallel(n_jobs=-1, verbose=10)(delayed(im_lcn_vec)(i, D, D,
                                                                   sigmas) \
                                           for i in test_set_t[0][:,1024:2048]))
    Xtest3 = np.asarray(Parallel(n_jobs=-1, verbose=10)(delayed(im_lcn_vec)(i, D, D,
                                                                        sigmas) \
                                           for i in test_set_t[0][:,2048:3072]))   

    Xvalid1 = np.asarray(Parallel(n_jobs=-1, verbose=10)(delayed(im_lcn_vec)(i, D, D,
                                                                   sigmas) \
                                           for i in valid_set_t[0][:,:1024]))
    Xvalid2 = np.asarray(Parallel(n_jobs=-1, verbose=10)(delayed(im_lcn_vec)(i, D, D,
                                                                   sigmas) \
                                           for i in valid_set_t[0][:,1024:2048]))
    Xvalid3 = np.asarray(Parallel(n_jobs=-1, verbose=10)(delayed(im_lcn_vec)(i, D, D,
                                                                        sigmas) \
                                           for i in valid_set_t[0][:,2048:3072]))   
   
    Xtrain = np.concatenate([Xtrain1, Xtrain2], axis=1)
    train_set_t[0] = np.concatenate([Xtrain, Xtrain3], axis=1)

    Xvalid = np.concatenate([Xvalid1, Xvalid2], axis=1)
    valid_set_t[0] = np.concatenate([Xvalid, Xvalid3], axis=1)

    Xtest = np.concatenate([Xtest1, Xtest2], axis=1)
    test_set_t[0] = np.concatenate([Xtest, Xtest3], axis=1)

    save_dataset(train_set_t, valid_set_t, test_set_t, 'cifar10_lcn')
    return train_set_t, valid_set_t, test_set_t


    
def dist2hy(x,y):
    '''Distance matrix computation
    Hybrid of the two, switches based on dimensionality
    '''
    #if x.eval().shape[1]<5:  #If the dimension is small
    #    d = T.zeros_like(T.dot(x, y.T))
    #    #d = np.zeros((x.shape[0],y.shape[0]),dtype=x.dtype)
    #    for i in xrange(x.eval().shape[1]):
    #        diff2 = x[:,i,None] - y[:,i]
    #        diff2 **= 2
    #        d += diff2
    #    #np.sqrt(d,d)
    #    return d

    #else:
    d = T.dot(x,y.T)
    d *= -2.0
    d += T.sum(x*x, axis=1).dimshuffle(0,'x')
    d += T.sum(y*y, axis=1)
    # Rounding errors occasionally cause negative entries in d
    d = d * T.cast(d>0,theano.config.floatX)
    #d[d<0] = 0
    # in place sqrt
    #np.sqrt(d,d)
    return T.sqrt(d)

def save_the_weight(x,fname):
    f = file(fname+'.save', 'wb')
    cPickle.dump(x, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


