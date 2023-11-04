# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch

class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.num_layers = lstm_num_layers
        self.num_hidden = lstm_num_hidden
        self.device = device
        #self.embedding = nn.Embedding(vocabulary_size, vocabulary_size) #map characters eq a = [1,0..0] to other dim vectors

        self.lstm = nn.LSTM(input_size=vocabulary_size,
                            hidden_size=lstm_num_hidden,
                            num_layers=lstm_num_layers,
                            batch_first=True)

        self.linear = nn.Linear(lstm_num_hidden,vocabulary_size)


    def forward(self, x, h= None,c = None ):
        # Implementation here...
        #init hidden and cell state with zeros

        if h == None:
            h0 = torch.zeros(self.num_layers, self.batch_size, self.num_hidden ).to(self.device) #LxBxH
            c0 = torch.zeros(self.num_layers, self.batch_size, self.num_hidden ).to(self.device)
        else:
            h0 = h
            c0 = c

        out, (ht,ct) = self.lstm(x,(h0,c0))  #x dim BxTxVocsize; output BxTxH

        p = self.linear(out)

        return p, (ht, ct)