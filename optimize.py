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
import pylab
import cPickle

import theano
import theano.tensor as T
import theano.tensor.signal.conv 
from theano.tensor.shared_randomstreams import RandomStreams
 
from collections import OrderedDict


class GraddescentMinibatch(object):
    """ Gradient descent trainer class. """

    def __init__(self, hyper_params):

        self.batch_sz, self.epsilon, self.momentum, self.lam\
                                                                = hyper_params

    def batch_update(self, model, train_set, valid_set):

        update_grads = []; updates_mom = []; deltaWs = {}
        X = T.fmatrix('X');  y = T.ivector('y'); 
        mom = T.scalar('mom'); i = T.iscalar('i'); lr = T.scalar('lr');

        params = model.params
        cost = model.cost(X,y) + self.lam * model.weight_decay()
        cost_test = model.cost(X,y) 
        grads = T.grad(cost, params)


        #Update momentum   
        updates = OrderedDict()  

        #Update momentum
        for param in model.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            deltaWs[param] = theano.shared(init)

        for param in model.params:
            updates_mom.append((param, param + deltaWs[param] * \
                            T.cast(mom, dtype=theano.config.floatX)))       


        for param, gparam in zip(params, gparams):

            deltaV = T.cast(mom, dtype=theano.config.floatX)\
                    * deltaWs[param] - gparam * T.cast(lr, dtype=theano.config.floatX)     #new momentum

            update_grads.append((deltaWs[param], deltaV))


        #for p, g in zip(params, grads):
        #    v = theano.shared(p.get_value()*0.)
        #    updates[v] = mom * v - T.cast(lr, dtype=theano.config.floatX) * g
        #    updates[p] = p + updates[v]


        train_update    = theano.function([i, theano.Param(lr,default=self.epsilon),\
                theano.Param(mom,default=self.momentum)], outputs=cost, updates=updates,\
                                givens={ X:train_set[0][i*self.batch_sz:(i+1)*self.batch_sz], \
                                         y:train_set[1][i*self.batch_sz:(i+1)*self.batch_sz]})

        get_valid_cost   = theano.function([i], outputs=cost_test,\
                    givens={ X:valid_set[0][i*self.batch_sz:(i+1)*self.batch_sz], \
                             y:valid_set[1][i*self.batch_sz:(i+1)*self.batch_sz]})

        return train_update, get_valid_cost

    def stochastic_update(self, model, train_set, valid_set):

        update_grads = []; updates_mom = []; deltaWs = {}
        X = T.fmatrix('X');  y = T.ivector('y'); 
        mom = T.scalar('mom'); i = T.iscalar('i'); lr = T.scalar('lr');
        perm1 = T.lvector('perm1'); 

        params = model.params
        cost = model.cost(X,y) + self.lam * model.weight_decay()
        cost_test = model.cost(X,y) 
        grads = T.grad(cost, params, consider_constant=[perm1])


        #Update momentum   
        updates = OrderedDict()  

        for p, g in zip(params, grads):
            v = theano.shared(p.get_value()*0.)
            updates[v] = mom * v - T.cast(lr, dtype=theano.config.floatX) * g
            updates[p] = p + updates[v]


        train_update    = theano.function([perm1, theano.Param(lr,default=self.epsilon),\
                theano.Param(mom,default=self.momentum)], outputs=cost, updates=updates,\
                                givens={ X:train_set[0][perm1], \
                                         y:train_set[1][perm1]})

        get_valid_cost   = theano.function([i], outputs=cost_test,\
                    givens={ X:valid_set[0][i*self.batch_sz:(i+1)*self.batch_sz], \
                             y:valid_set[1][i*self.batch_sz:(i+1)*self.batch_sz]})

        return train_update, get_valid_cost


    def get_error(self, model, valid_set, test_set):

        X = T.fmatrix('X');  y = T.ivector('y'); 
        error_rate = model.errors(X, y)
        return theano.function([], error_rate, givens={ X:valid_set[0], y:valid_set[1]})

           
    def compute_grad(self, model, train_set):
        
        update_grads = []; updates_mom = []; deltaWs = {}
        X = T.fmatrix('X');  y = T.ivector('y'); 
        mom = T.scalar('mom'); i = T.iscalar('i'); lr = T.scalar('lr');
        perm1 = T.lvector('perm1'); 

        params = model.params
        cost = model.cost(X,y) + self.lam * model.weight_decay()
        cost_test = model.cost(X,y) 
        gparams = T.grad(cost, params, consider_constant=[perm1])


        get_grad = theano.function([i], gparams, \
                givens={ X:train_set[0][i*self.batch_sz:(i+1)*self.batch_sz], \
                         y:train_set[1][i*self.batch_sz:(i+1)*self.batch_sz]})

        return get_grad


