"""
This module implements a bidirectional LSTM in PyTorch.
You should fill in code into indicated sections.
Date: 2020-11-09
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch

class biLSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(biLSTM, self).__init__()
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.device = device        

        # initial hidden state
        self.hi = nn.Parameter(torch.zeros(batch_size, hidden_dim), requires_grad=False)
        self.ci = nn.Parameter(torch.zeros(batch_size, hidden_dim), requires_grad=False)
        #final hidden state
        self.hf = nn.Parameter(torch.randn(batch_size, hidden_dim), requires_grad=False)
        self.cf = nn.Parameter(torch.randn(batch_size, hidden_dim), requires_grad=False)
        
        self.forwardCell = LSTMCell(seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device)
        self.backwardCell = LSTMCell(seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device)
        

        self.Wph = nn.Parameter(torch.randn(hidden_dim*2, num_classes))
        nn.init.kaiming_normal_(self.Wph)

        self.bp = nn.Parameter(torch.zeros(1, num_classes))
        
        self.softmax = nn.Softmax(dim=1)

        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        ctf = self.ci
        htf = self.hi
        ctb = self.cf
        htb = self.hf
        for seq_el in range(self.seq_length-1):
            ctf, htf = self.forwardCell(x[:,seq_el].reshape(self.batch_size,self.input_dim),ctf,htf)
            ctb, htb = self.backwardCell(x[:,(self.seq_length-2)-seq_el].reshape(self.batch_size,self.input_dim),ctb,htb)
        #for seq_el in range(self.seq_length-1):
        #    ctb, htb = self.backwardCell(x[:,(self.seq_length-2)-seq_el].reshape(128,1),ctb,htb)
        
        H = torch.cat((htf,htb),dim=1)
        pt = H @ self.Wph + self.bp
        out = self.softmax(pt)
        #print(self.Wph.shape, ht.shape,out.shape)
        return out
        ########################
        # END OF YOUR CODE    #
        #######################


class LSTMCell(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(LSTMCell, self).__init__()

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.device = device


        #init state no grads requirement
        # initial hidden state
        #self.hi = nn.Parameter(torch.zeros(batch_size, hidden_dim), requires_grad=False)
        #self.ci = nn.Parameter(torch.zeros(batch_size, hidden_dim), requires_grad=False)

        #weight matrices
        self.Wgx = nn.Parameter(torch.randn(input_dim, hidden_dim)) # 
        self.Wgh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        #nn.init.kaiming_normal_(self.Wgx)
        #nn.init.kaiming_normal_(self.Wgh)

        self.Wix = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.Wih = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        nn.init.kaiming_normal_(self.Wix)
        nn.init.kaiming_normal_(self.Wih)

        self.Wfx = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.Wfh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        nn.init.kaiming_normal_(self.Wfx)
        nn.init.kaiming_normal_(self.Wfh)

        self.Wox = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.Woh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        nn.init.kaiming_normal_(self.Wox)
        nn.init.kaiming_normal_(self.Woh)

        

        #biases vectors
        self.bg = nn.Parameter(torch.zeros(1, hidden_dim))
        self.bi = nn.Parameter(torch.zeros(1, hidden_dim))
        self.bf = nn.Parameter(torch.zeros(1, hidden_dim))
        self.bo = nn.Parameter(torch.zeros(1, hidden_dim))
        
        #activation functions
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x, c, h):
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        
        gt = self.tanh( x @ self.Wgx + h @ self.Wgh + self.bg)
        it = self.sigmoid(x @ self.Wix + h @ self.Wih + self.bi)
        ft = self.sigmoid(x @ self.Wfx + h @ self.Wfh + self.bf)
        ot = self.sigmoid(x @ self.Wox + h @ self.Woh + self.bo)
        ct = gt * it + c * ft
        ht = self.tanh(ct) * ot
        #pt = ht @ self.Wph + self.bp
        #out = self.softmax(pt)
        #print(self.Wph.shape, ht.shape,out.shape)
        return ct, ht
        ########################
        # END OF YOUR CODE    #
        #######################
