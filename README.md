# Learning a Metric for Class-Conditional KNN

Python (Theano) implementation of Learning a Metric for Class-Conditional KNN code provided 
by Daniel Jiwoong Im and Graham W Taylor.
The codes include experiments on hodge decomposition, in particular convservative components (for now),
and vector field deformations in 2D. For more information, see 

```bibtex
@article{Im2016ccml,
    title={Learning a Metric for Class-Conditional KNN},
    author={Im, Daniel Jiwoong and Taylor, Graham W.},
    journal={International Joint Conference on Neural Networks (To appear)},
    year={2016}
}
```

If you use this in your research, we kindly ask that you cite the above workshop paper


## Dependencies
Packages
* [numpy](http://www.numpy.org/)
* [Theano ('0.7.0.dev-725b7a3f34dd582f9aa2071a5b6caedb3091e782')](http://deeplearning.net/software/theano/) 


## How to run
Entry code for one-bit flip and factored minimum probability flow for mnist data are 
```
    - /test_ccml2_mnist.py
    - /test_ccml2_norb.py
```

