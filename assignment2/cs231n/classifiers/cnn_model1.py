import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNetModel1(object):
  """
  A three convolutional network with the following architecture:

  [conv-relu-pool]xN - conv - relu - [affine]xM - [softmax or SVM]
  # [conv-relu-pool]XN - [affine]XM - [softmax or SVM]
  # [conv-relu-conv-relu-pool]xN - [affine]xM - [softmax or SVM]

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    F = num_filters

    # Conv relu layer
    # Input size : (N,C,H,W)
    # Output size : (N,F,Hc,Wc)
    F_H = filter_size
    F_W = filter_size
    S = 1 #stride
    P = (filter_size - 1) / 2 # padding
    Hc = 1 + (H + 2 * P - F_H) / S
    Wc = 1 + (W + 2 * P - F_W) / S

    self.params['W1'] = weight_scale * np.random.randn(F,C,F_H,F_W)
    self.params['b1'] = weight_scale * np.random.randn(F)

    # N="mini-batch"
    # print "After Conv Relu: "
    # print "Input :"+ str((N,C,H,W))
    # print "W1: "+str((F,C,F_H,F_W))
    # print "Ouput :"+ str((N,F,Hc,Wc))

    # Pool layer : 2*2
    # Input : (N,F,Hc,Wc)
    # Ouput : (N,F,Hp,Wp)

    width_pool = 2
    height_pool = 2
    stride_pool = 2
    Hp = (Hc - height_pool) / stride_pool + 1
    Wp = (Wc - width_pool) / stride_pool + 1

    # print "After Pool: "
    # print "Input :"+ str((N,F,Hc,Wc))
    # print "Ouput :"+ str((N,F,Hp,Wp))

    # Conv relu layer
    # Input size : (N,F,Hp,Wp)
    # Output size : (N,F,Hc,Wc)
    F_H = filter_size
    F_W = filter_size
    S = 1 #stride
    P = (filter_size - 1) / 2 # padding
    Hc = 1 + (Hp + 2 * P - F_H) / S
    Wc = 1 + (Wp + 2 * P - F_W) / S

    self.params['W2'] = weight_scale * np.random.randn(F,F,F_H,F_W)
    self.params['b2'] = weight_scale * np.random.randn(F)

    # print "After Conv Relu: "
    # print "Input :"+ str((N,F,Hp,Wp))
    # print "W2: "+str((F,F,F_H,F_W))
    # print "Ouput :"+ str((N,F,Hc,Wc))

    # Output Affine layer
    # parameters (F*Hc*Wc,num_classes)
    # Input: (N,F*Hc*Wc)
    # Output: (N,num_classes)

    self.params['W3'] = weight_scale * np.random.randn(F*Hc*Wc, num_classes)
    self.params['b3'] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # print "======================="
    # print "X.shape: "+str(X.shape)

    # Forward into the conv_relu_pool input layer1
    out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    # print "W1.shape: "+str(W1.shape)
    # print "out1.shape: "+str(out1.shape)

    # Forward into the conv_relu layer2
    out2, cache2 = conv_relu_forward(out1, W2, b2, conv_param)
    # print "W2.shape: "+str(W2.shape)
    # print "out2.shape: "+str(out2.shape)

    # Forward into the affine output layer3
    out3, cache3 = affine_forward(out2, W3, b3)
    scores = out3
    # print "W3.shape: "+str(W3.shape)
    # print "out3.shape: "+str(out3.shape)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################

    # Computing of the loss
    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * (np.sum(W1**2)+np.sum(W2**2)+np.sum(W3**2))
    loss = data_loss + reg_loss

    dout3, grads['W3'], grads['b3'] = affine_backward(dscores, cache3)
    dout2, grads['W2'], grads['b2'] = conv_relu_backward(dout3, cache2)
    dout1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout2, cache1)

    grads['W1'] += self.reg * W1
    grads['W2'] += self.reg * W2
    grads['W3'] += self.reg * W3

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass
