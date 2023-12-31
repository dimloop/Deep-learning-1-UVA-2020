"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        neurons =[n_inputs] + n_hidden + [n_classes] #list of neurons in each layer
        self.Linear_layers = [LinearModule(neurons[i],neurons[i+1]) for i in range(0,len(neurons)-1)]
        self.actF = ELUModule()
        
        self.softmax = SoftMaxModule()
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        for layer in self.Linear_layers[:-1]:
            x = layer.forward(x)
            x = self.actF.forward(x)

        out = self.Linear_layers[-1].forward(x)
        out = self.softmax.forward(out)
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        dout = self.softmax.backward(dout)
        dout = self.Linear_layers[-1].backward(dout)
        for layer in reversed(self.Linear_layers[:-1]):
            dout = self.actF.backward(dout)
            dout = layer.backward(dout)
        ########################
        # END OF YOUR CODE    #
        #######################

        return
