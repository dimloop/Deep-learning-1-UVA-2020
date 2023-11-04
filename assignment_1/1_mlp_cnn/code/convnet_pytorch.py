"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch

class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """
    
    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.
        
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
          
        
        TODO:
        Implement initialization of the network.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        super(ConvNet, self).__init__()
        
        def conv(input, out):
            return nn.Sequential(
                nn.Conv2d(input, out, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        
        def PreAct(input, out):
            return nn.Sequential(
                nn.BatchNorm2d(input),
                nn.ReLU(),
                nn.Conv2d(input, out, kernel_size=3, stride=1, padding=1)
            )

        self.conv_net= nn.Sequential(
            conv(n_channels, 64), #conv0
            PreAct(64, 64), #PreAct1
            conv(64, 128), #conv1
            PreAct(128, 128), #PreAct2a
            PreAct(128, 128), #PreAct2b
            conv(128, 256), #conv2
            PreAct(256, 256), #PreAct3a
            PreAct(256, 256), #PreAct3b
            conv(256, 512), #conv3
            PreAct(512, 512), #PreAct4a
            PreAct(512, 512), #PreAct4b
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            PreAct(512, 512), #PreAct5a
            PreAct(512, 512), #PreAct5b
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten()
            )
        
        self.linear = nn.Linear(512, n_classes, bias=True)

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
        out = self.conv_net(x)
        #out = nn.Flatten(out)
        #out = out.view(out.size(0), -1)
        out = self.linear(out)
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out

if __name__ == '__main__':
    cv = ConvNet(3,10)
    torch.manual_seed(42)
    test = torch.randn(10, 3,32,32)
    print(cv(test))