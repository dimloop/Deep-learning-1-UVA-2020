################################################################################
# MIT License
#
# Copyright (c) 2020 Phillip Lippe
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2020
# Date Created: 2020-11-22
################################################################################

import torch
import torch.nn as nn
import numpy as np


class GeneratorMLP(nn.Module):

    def __init__(self, z_dim=32, hidden_dims=[128, 256, 512, 1024], output_shape=[1, 28, 28], dp_rate=0.1):
        """
        Generator network with linear layers, LeakyReLU activations (alpha=0.2) and dropout. The output layer
        uses a Tanh activation function to scale the output between -1 and 1.

        Inputs:
            z_dim - Dimensionality of the latent input space. This is the number of neurons of the input layer
            hidden_dims - List of dimensionalities of the hidden layers in the network.
                          The NN should have the same number of hidden layers as the length of the list.
            output_shape - Shape of the output image excluding the batch size. The number of output neurons
                           of the NN must be the product of the shape elements.
            dp_rate - Dropout probability to apply after every linear layer except the output.
        """
        super().__init__()
        # You are allowed to experiment with the architecture and change the activation function, normalization, etc.
        # However, the default setup is sufficient to generate fine images and gain full points in the assignment.
        # The default setup is a sequence of Linear, Dropout, LeakyReLU (alpha=0.2)
        alpha = 0.2
        self.output_shape = output_shape
        layers = [z_dim]+hidden_dims+[output_shape[0]*output_shape[1]*output_shape[2]]

        self.generator = nn.ModuleList()
        for i in range(len(layers)-1):
          if i < len(layers)-2:
            self.generator.append(nn.Linear(layers[i], layers[i+1]))
            self.generator.append(nn.Dropout(dp_rate, inplace=True))
            self.generator.append(nn.LeakyReLU(alpha, inplace=True))
          else:
            self.generator.append(nn.Linear(layers[i], layers[i+1]))
            self.generator.append(nn.Tanh())
        self.generator = nn.Sequential(*self.generator)

    def forward(self, z):
        """
        Inputs:
            z - Input batch of latent vectors. Shape: [B,z_dim]
        Outputs:
            x - Generated image of shape [B,output_shape[0],output_shape[1],output_shape[2]]
        """
        x = self.generator(z)
        return x.reshape(z.shape[0], self.output_shape[0], self.output_shape[1], self.output_shape[2])

    @property
    def device(self):
        """
        Property function to get the device on which the generator is
        """
        return next(self.parameters()).device


class DiscriminatorMLP(nn.Module):

    def __init__(self, input_dims=784, hidden_dims=[512, 256], dp_rate=0.3):
        """
        Discriminator network with linear layers, LeakyReLU activations (alpha=0.2) and dropout.

        Inputs:
            input_dims - Number of input neurons/pixels. For MNIST, 28*28=784
            hidden_dims - List of dimensionalities of the hidden layers in the network. 
                          The NN should have the same number of hidden layers as the length of the list.
            dp_rate - Dropout probability to apply after every linear layer except the output.
        """
        super().__init__()
        # You are allowed to experiment with the architecture and change the activation function, normalization, etc.
        # However, the default setup is sufficient to generate fine images and gain full points in the assignment.
        # The default setup is the same as the generator: a sequence of Linear, Dropout, LeakyReLU (alpha=0.2)
        alpha = 0.2
        self.discriminator = nn.Sequential(
            nn.Linear(input_dims, hidden_dims[0]),
            nn.Dropout(dp_rate, inplace = True),
            nn.LeakyReLU(alpha, inplace=True),
            nn.Linear(hidden_dims[0],  hidden_dims[1]),
            nn.Dropout(dp_rate, inplace = True),
            nn.LeakyReLU(alpha, inplace=True),
            nn.Linear(hidden_dims[1], 1)
        )

    def forward(self, x):
        """
        Inputs:
            x - Batch of input images. Shape: [B,C,H,W]
        Outputs:
            preds - Predictions whether a specific image is fake (0) or real (1).
                    Note that this should be a logit output *without* a sigmoid applied on it.
                    Shape: [B,1]
        """
        preds = self.discriminator(x)
        return preds
