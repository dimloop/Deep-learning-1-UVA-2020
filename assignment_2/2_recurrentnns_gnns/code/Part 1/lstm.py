"""
This module implements a LSTM model in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device = 'cpu'):

        super(LSTM, self).__init__()
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.device = device


        #init state no grads requirement
        # initial hidden state
        self.hi = nn.Parameter(torch.zeros(batch_size, hidden_dim), requires_grad=False)
        self.ci = nn.Parameter(torch.zeros(batch_size, hidden_dim), requires_grad=False)

        #weight matrices
        self.Wgx = nn.Parameter(torch.randn(input_dim, hidden_dim)) # 
        self.Wgh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        #nn.init.kaiming_normal_(self.Wgx, mode='fan_in', nonlinearity='relu')
        #nn.init.kaiming_normal_(self.Wgh, mode='fan_in', nonlinearity='relu')

        self.Wix = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.Wih = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        nn.init.kaiming_normal_(self.Wix, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.Wih, mode='fan_in', nonlinearity='relu')

        self.Wfx = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.Wfh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        nn.init.kaiming_normal_(self.Wfx, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.Wfh, mode='fan_in', nonlinearity='relu')

        self.Wox = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.Woh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        nn.init.kaiming_normal_(self.Wox, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.Woh, mode='fan_in', nonlinearity='relu')

        self.Wph = nn.Parameter(torch.randn(hidden_dim, num_classes))
        nn.init.kaiming_normal_(self.Wph, mode='fan_in', nonlinearity='relu')

        #biases vectors
        self.bg = nn.Parameter(torch.randn(1, hidden_dim))
        self.bi = nn.Parameter(torch.randn(1, hidden_dim))
        self.bf = nn.Parameter(torch.randn(1, hidden_dim))
        self.bo = nn.Parameter(torch.randn(1, hidden_dim))
        self.bp = nn.Parameter(torch.randn(1, num_classes))
        
        #activation functions
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        #print(x.shape, self.Wgx.shape)
        #emb = nn.Embedding(24,self.input_dim)
        #print(x.type(torch.int64))
        #x.type(torch.int64)
        #print(emb(x).shape, emb(x).reshape(256,24,3).shape)
        #x = emb(x).reshape(self.batch_size,self.seq_length,self.input_dim)
       # x.permute(0, 2, 1)
        ht = self.hi
        ct = self.ci
        
       # print(self.Wgx.shape, x[:,4].reshape(128,1)@self.Wgx)
        #print( "AA",(ht @ self.Wgh).shape,(x[:, 1, :] @ self.Wgx).shape, self.bg.shape )
        for seq_el in range(self.seq_length-1):
            x_ = x[:, seq_el].reshape(128,1)
            gt = self.tanh( x_ @ self.Wgx + ht @ self.Wgh + self.bg)
            it = self.sigmoid(x_ @ self.Wix + ht @ self.Wih + self.bi)
            ft = self.sigmoid(x_ @ self.Wfx + ht @ self.Wfh + self.bf)
            ot = self.sigmoid(x_ @ self.Wox + ht @ self.Woh + self.bo)
            ct = gt * it + ct * ft
            ht = self.tanh(ct) * ot
        pt = ht @ self.Wph + self.bp
        out = self.softmax(pt)
        #print(self.Wph.shape, ht.shape,out.shape)
        return out
        ########################
        # END OF YOUR CODE    #
        #######################

if __name__ == "__main__":
    test = torch.randint(10,(256,24,1))
    print(test.shape)    
    emb = torch.nn.Embedding(24, 3)
    print(emb)
    print(emb(test).shape)
    word_conversion = {"hey": 0, "there": 1}
    embeddings = nn.Embedding(2, 3)
    lookup = torch.tensor([word_conversion["hey"]], dtype=torch.long)
    hey_embeddings = embeddings(lookup)
    print(hey_embeddings, lookup)
    model = LSTM(24,3,256,2,256)
    print(model(test).sum(1))