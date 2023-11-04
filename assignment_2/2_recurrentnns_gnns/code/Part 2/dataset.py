# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch.utils.data as data


class TextDataset(data.Dataset):

    def __init__(self, filename, seq_length):
        assert os.path.splitext(filename)[1] == ".txt"
        self._seq_length = seq_length
        self._data = open(filename, 'r').read()
        self._chars = sorted(list(set(self._data)))
        self._data_size, self._vocab_size = len(self._data), len(self._chars)
        print("Initialize dataset with {} characters, {} unique.".format(
            self._data_size, self._vocab_size))
        self._char_to_ix = {ch: i for i, ch in enumerate(self._chars)} #char to numbers
        #print(self._char_to_ix)
        #print(self._chars)
        self._ix_to_char = {i: ch for i, ch in enumerate(self._chars)} #numbers to char
        #print(self._ix_to_char)
        #print("aaaa",self._ix_to_char[32])

        self._offset = 0

    #generates random elements of the text with length equal seq length
    def __getitem__(self, item):
        offset = np.random.randint(0, len(self._data)-self._seq_length-2)
        inputs = [self._char_to_ix[ch] for ch in self._data[offset:offset+self._seq_length]]
        targets = [self._char_to_ix[ch] for ch in self._data[offset+1:offset+self._seq_length+1]]
        return inputs, targets

    def convert_to_string(self, char_ix):
        return ''.join(self._ix_to_char[ix] for ix in char_ix)

    def __len__(self):
        return self._data_size

    @property
    def vocab_size(self):
        return self._vocab_size

###### ignore############
if __name__ == "__main__":    
    from torch.utils.data import DataLoader
    import torch
    import torch.nn as nn
    from model import TextGenerationModel

    #obj = TextDataset("./assets/book_EN_democracy_in_the_US.txt",15)
    txt_file = "./assets/book_EN_grimms_fairy_tails.txt"
    T=2
    batch_size=3
    #print(obj[4],obj[4])
    #print(obj.convert_to_string(obj[4][0]))
    #print(obj.vocab_size)
    
    dataset = TextDataset(txt_file, T)  # fixme
    print("dataset len ",len(dataset)) #total characters
    data_loader = DataLoader(dataset, batch_size,drop_last=True)
    model = TextGenerationModel(batch_size, T, dataset.vocab_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cpu')
    
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        #print(batch_inputs)
        dat = batch_inputs
        tar = batch_targets
        #print(torch.nn.functional.one_hot(batch_inputs[0], dataset.vocab_size))

        if step==0:
            break
    
    #dat, _ = dataset[0]
    print("data example",dat, tar)

    dat = torch.stack(dat).t()
    #print("mmmm", model(dat))
    
    tar = torch.stack(tar).t()
    print("new data", dat)#, "new target", tar)
    #print(dat.transpose(0,1))
    #conver to one hot
    data = torch.nn.functional.one_hot(dat,dataset.vocab_size).type(torch.FloatTensor)
    targets = torch.nn.functional.one_hot(tar,dataset.vocab_size).type(torch.FloatTensor)
    print(data.shape)
    print("one hot", data)
    print(torch.nn.functional.one_hot(torch.tensor([1]),dataset.vocab_size).type(torch.FloatTensor))
    #convert to BxLxHdim
    #data = torch.zeros(batch_size, T, dataset.vocab_size)
   
    #for d in dat:
    #    print(d)
    #dat=[torch.nn.functional.one_hot(torch.tensor(d),dataset.vocab_size).type(torch.FloatTensor) for d in dat]
    
    #print(torch.nn.functional.one_hot(torch.tensor(dat),dataset.vocab_size))
    #print(dat[0])
    #print(torch.randn(1, 3, 10))
    aa = model(data)
    #print("aa",model(data).shape)
    print(targets.shape)
    import random
    print(random.choice(dataset._chars))
    print(random.choice(dataset._chars))
    #print(torch.argmax(aa, dim=2))
    #predictions  = torch.argmax(aa, dim=2)
    #print(tar)
    #print(aa.size(1))
    #correct = (predictions == tar).sum().item()
    #accuracy = correct / (aa.size(0)*aa.size(1)) # fixme
    #print(predictions == tar)
    #print(aa.shape)
    #for step, (batch_inputs, batch_targets) in enumerate(data_loader):
    #    print(batch_inputs)
        #print(torch.nn.functional.one_hot(batch_inputs[0], dataset.vocab_size))

    #    if step==0:
    #        break
    #loss = nn.CrossEntropyLoss()
    #print(loss(aa,targets))
    #print(torch.argmax(aa, dim=2).shape)
    #print(torch.nn.CrossEntropyLoss(aa[0], targets[0]))
    def one_hot(x, vocab_size):
        return torch.nn.functional.one_hot(x,vocab_size).type(torch.FloatTensor)
    print("--------------------------------")
    chara = "A"
    print(torch.stack([torch.tensor(dataset._char_to_ix[chara])]))
    ch_to_m = one_hot(torch.stack([torch.tensor(dataset._char_to_ix[chara])]).t(),dataset.vocab_size)
    
    print(ch_to_m)
    print(dataset._char_to_ix[chara])
    print("qqqqq",torch.nn.functional.one_hot(torch.stack([torch.tensor(dataset._char_to_ix[chara])]).t()).type(torch.FloatTensor).shape)
    h0 = torch.zeros(2, 256 )
    c0 = torch.zeros(2, 256 )
    sf =  nn.Softmax(dim=1)
    #print(torch.nn.functional.one_hot(torch.tensor(dataset._char_to_ix[chara]),dataset.vocab_size).type(torch.FloatTensor).shape)
    #out = ch_to_m.reshape(1,dataset.vocab_size)
    #print(out.shape)
    #print(model(test,h0,c0))
    #txt = ''
    #with torch.no_grad():
    #    for i in range(1):
     #       out , (h0, c0) = model(out,h0,c0)
     #       print(out.shape)
     #       out = torch.argmax(sf(out))
      #      print(out)
            #print(outt, torch.argmax(outt))
      #      txt = txt + dataset.convert_to_string([out.item()])
      #      out  = torch.nn.functional.one_hot(out,dataset.vocab_size).type(torch.FloatTensor).reshape(1,dataset.vocab_size)
    #print(txt)
    #print(out)
    #print(h0.shape)
    #rnd_char = np.random.choice(list(map(chr, range(97, 123)))).upper()
    #prev = torch.zeros(dataset.vocab_size)
    #prev[dataset._chars.index(rnd_char)] = 1
    #prev = prev.view(1,1,-1) #dim: B x T x D
    #print(prev.shape)
    #for i in range(30-1):
    #    gen_y = model(prev) #dim: B x T x C
    #    char = torch.zeros(dataset.vocab_size)
    #    softm = torch.softmax(1*gen_y[0,-1,:],0).squeeze() #temperature included
#   #                    char[np.random.choice(np.arange(dataset.vocab_size),p=np.array(softm.cpu()))] = 1
    #    char[torch.argmax(softm)] = 1 #greedy, uncomment prev line for random
    #    prev = torch.cat([prev, char.view(1,1,-1)],1)
    #txt = dataset.convert_to_string(torch.argmax(prev,2).squeeze().tolist())
    #print(torch.argmax(prev,2).squeeze().tolist())
    #print(txt)
    #import random
    #print(random.choice(dataset._chars))
    #print(len(dataset)/64)