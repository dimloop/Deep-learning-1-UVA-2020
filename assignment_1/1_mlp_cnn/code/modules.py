"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """
    
    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.
    
        Args:
          in_features: size of each input sample
          out_features: size of each output sample
    
        TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.
    
        Also, initialize gradients with zeros.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        weight = np.random.normal(0,.0001,(in_features, out_features))
        bias = np.zeros([1, out_features])
        grads = np.zeros([in_features, out_features])

        self.params = {'weight': weight, 'bias': bias}
        self.grads = {'weight': grads, 'bias': bias}        


        ########################
        # END OF YOUR CODE    #
        #######################
    
    def forward(self, x):
        """
        Forward pass.
    
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        TODO:
        Implement forward pass of the module.
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = x
        out = x @ self.params["weight"]  + self.params["bias"]
        self.out = out
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
    
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.grads["weight"] = self.x.T @ dout
        self.grads["bias"] = np.ones([1,len(dout[:,0])]) @ dout
        
        dx = dout @ self.params["weight"].T

        ########################
        # END OF YOUR CODE    #
        #######################
        return dx



class SoftMaxModule(object):
    """
    Softmax activation module.
    """
    
    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        b = np.max(x, axis = 1)
        b = b.reshape(len(b), 1)@np.ones([1, len(x[0,:])])
        out = np.exp(x-b)
        norm = out.sum(axis=1).reshape(len(b),1) @ np.ones([1, len(x[0,:])])
        out = out / norm
        self.out = out 
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        sec_term =(dout*self.out).sum(axis=1).reshape(len(self.out[:,1]),1)@np.ones([1,len(self.out[0,:])])

        dx = dout * self.out -  self.out * sec_term #*
        #print(self.out*aa)
        #hard to do vectorized
        #dx = np.zeros(np.shape(dout))
        #for i in range(len(dout[:,0])):

        #  self_out = self.out[i,:].reshape((len(self.out[i,:]),1))
        #  dsoftm = np.diag(self_out.squeeze()) - self_out@self_out.T
        #  dx[i,:] = dout[i,:] @ dsoftm
    

        ########################
        # END OF YOUR CODE    #
        #######################
        
        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """
    
    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
    
        TODO:
        Implement forward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        #out = np.mean(np.sum(-np.multiply(y,np.log(x+1e-30)), axis=1))
        out = - np.mean( np.sum(y * np.log(x), axis = 1))
        #out = 1
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.
    
        TODO:
        Implement backward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        #1/batchsize
        dx = - 1/len(x[:,0]) * y/x
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return dx


class ELUModule(object):
    """
    ELU activation module.
    """
    
    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.x = np.copy(x)
        out = np.copy(x)
        out[out<0] = np.exp(out[out<0] )-1          
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """
        #print(dout)
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        delu = self.x
        delu[delu >= 0] = 1
        delu[delu < 0] = np.exp(delu[delu < 0])
        #print("der",delu )
        #print("out",dout)
        dx = dout*delu
        ########################
        # END OF YOUR CODE    #
        #######################
        return dx


if __name__=="__main__":
    x_test = np.array([[1.,2.,3.],[1.,2.,3.]])
    #sf = SoftMaxModule()
    #print(sf.forward(x_test))
    #print(sf.backward(x_test))
    ce = CrossEntropyModule()
    print(ce.backward(x_test,x_test))
    #el = ELUModule()
    #print(x_test)
    #print(el.forward(x_test))
    #print(x_test)
    #print(el.backward(x_test))