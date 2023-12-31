"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
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
        super().__init__() #init base class nn.module
        
        neurons =[n_inputs] + n_hidden + [n_classes] #list of neurons in each layer
        self.Linear_layers = nn.ModuleList([nn.Linear(neurons[i],neurons[i+1]) for i in range(0,len(neurons)-1)])
        self.actF = nn.ELU()
        #self.softmax = nn.Softmax()
        #print(self.Linear_layers)
        
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
            x = layer(x)
            x = self.actF(x)
        out = self.Linear_layers[-1](x)
        #out = self.softmax(out)    
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out



if __name__=="__main__":
  import numpy as np
  import torch
  print(1)
  obj = MLP(1,[2,3],2)
  x=np.array([1,2,3])
  t = torch.from_numpy(x).float().requires_grad_()
  #t.requires_grad_()
  print(x,t)
  import matplotlib.pyplot as plt
  plt.figure()
  plt.plot(x,x)
  plt.figure()
  plt.plot(x,x)
  plt.show()
