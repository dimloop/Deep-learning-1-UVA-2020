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

import os
import time
from datetime import datetime
import argparse
import sys

import numpy as np
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

###############################################################################

def one_hot(x, vocab_size):
    return torch.nn.functional.one_hot(x,vocab_size).type(torch.FloatTensor)

def accuracy(pred, targets):
    predictions = torch.argmax(pred, dim=2)
    correct = (predictions ==targets).sum().item()
    acc = correct / (pred.size(0)*pred.size(1)) # fixme
    return acc

def train(config, dict=None):
    def set_seed(seed):
      np.random.seed(seed)
      torch.manual_seed(seed)
      if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    #set_seed(45)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    #for data
    losses = []
    accuracies = []
    steps = []
    
    # Initialize the device which to run the model on
    device = torch.device(config.device)
    #print(device)
    
    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size,drop_last=True)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size,
                 config.lstm_num_hidden, config.lstm_num_layers, config.device)  # FIXME

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # FIXME
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate) # FIXME
    #epochs = int((config.train_steps)/(len(dataset)/config.batch_size))
    prev_steps = 0
    config.sample_every = int(config.train_steps/4)
    #print(epochs, len(dataset))
    #print(epochs)
    #print(len(dataset))
    
    if not os.path.exists(config.summary_path):
      os.mkdir(config.summary_path)
    if config.sample_method == "greedy":
      if config.complete_sentence == False:
        file = open('{}/{}_{}.txt'.format(config.summary_path, datetime.now().strftime("%Y-%m-%d"), config.sample_method), 'w', encoding='utf-8')
        file.write('{}\n'.format(config.sample_method))
        file.write('\n')
      else:
        file = open('{}/{}_{}_complete.txt'.format(config.summary_path, datetime.now().strftime("%Y-%m-%d"), config.sample_method), 'w', encoding='utf-8')
        file.write('{}\n'.format(config.sample_method))
        file.write('\n')

    elif config.sample_method == "random":
      if config.complete_sentence == False:
        file = open('{}/{}_{}_{}.txt'.format(config.summary_path, datetime.now().strftime("%Y-%m-%d"), config.sample_method, config.temperature), 'w', encoding='utf-8')
        file.write('{}, temperature = {} \n'.format(config.sample_method, config.temperature))
        file.write('\n')
      else:
        file = open('{}/{}_{}_{}_complete.txt'.format(config.summary_path, datetime.now().strftime("%Y-%m-%d"), config.sample_method, config.temperature), 'w', encoding='utf-8')
        file.write('{}, temperature = {} \n'.format(config.sample_method, config.temperature))
        file.write('\n')  
    
    
    #for epoch in range(epochs):
    while True:
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            model.train()
            step = prev_steps+step
            # Only for time measurement of step through network
            t1 = time.time()

            #######################################################
            # Move to GPU
            batch_inputs = one_hot(torch.stack(batch_inputs).t(), dataset.vocab_size).to(device)


            batch_targets = torch.stack(batch_targets).t().to(device)   # [batch_sizexTxVoc]
           
            h0 = torch.zeros(config.lstm_num_layers, config.batch_size,config.lstm_num_hidden).to(config.device) #BxLxH
            c0 = torch.zeros(config.lstm_num_layers, config.batch_size, config.lstm_num_hidden ).to(config.device)
            # Forward pass
            model.to(device)
            pred, _ = model(batch_inputs, h0,c0)

            #print(batch_inputs.shape)
            optimizer.zero_grad()
            #######################################################
            
            # Compute the loss, gradients and update network parameters
            loss = criterion(pred.permute(0,2,1), batch_targets)  # fixme
            loss.backward()
            acc = accuracy(pred, batch_targets) # fixme

            #torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)  # prevents maximum gradient problem

            optimizer.step()

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)


            if (step ) % config.print_every == 0:
            #if (step ) % config.sample_every == 0:
                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                        Examples/Sec = {:.2f}, "
                    "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                        config.train_steps, config.batch_size, examples_per_second,
                        acc, loss
                        ))

                losses.append(loss.item())
                accuracies.append(acc)
                steps.append(step)

            if step  % config.sample_every  == 0:
                print(step, config.sample_every)
                model.eval()
                
                if config.sample_method == "None":
                  continue
                
                if config.sample_method == "greedy":
                    if config.complete_sentence == False:
                      softmax = torch.nn.Softmax(dim=1).to(device)
                      txt = np.random.choice(dataset._chars)
                      # Generate some sentences by sampling from the model
                      out = one_hot(torch.stack([torch.tensor(dataset._char_to_ix[txt])]).t(),dataset.vocab_size).to(device)
                      h0 = torch.zeros(config.lstm_num_layers, config.lstm_num_hidden ).to(device)
                      c0 = torch.zeros(config.lstm_num_layers, config.lstm_num_hidden ).to(device)
                      for i in range(config.gen_seq_length):
                        out , (h0, c0) = model(out,h0,c0)
                        out = torch.argmax(softmax(out))
                        txt = txt + dataset.convert_to_string([out.item()])
                        out = one_hot(torch.stack([torch.tensor(out.item())]).t(),dataset.vocab_size).to(device)
                    
                      print("TEXT "+ config.sample_method+": ", txt)
                      file.write('Training Step: {}\n'.format(step))
                      file.write('--------------\n')
                      file.write('{}\n'.format(txt))
                      file.write('\n')
                    else:
                      softmax = torch.nn.Softmax(dim=1).to(device)
                      txt = config.input_seq
                      # Generate some sentences by sampling from the model
                      h0 = torch.zeros(config.lstm_num_layers, config.lstm_num_hidden ).to(device)
                      c0 = torch.zeros(config.lstm_num_layers, config.lstm_num_hidden ).to(device)
                      for char in txt:
                        out = one_hot(torch.stack([torch.tensor(dataset._char_to_ix[char])]).t(),dataset.vocab_size).to(device)
                        _ , (h0, c0) = model(out,h0,c0)
                      out = txt[-1]  
                      out = one_hot(torch.stack([torch.tensor(dataset._char_to_ix[out])]).t(),dataset.vocab_size).to(device)

                      for i in range(config.gen_seq_length):
                        out , (h0, c0) = model(out,h0,c0)
                        out = torch.argmax(softmax(out))
                        txt = txt + dataset.convert_to_string([out.item()])
                        out = one_hot(torch.stack([torch.tensor(out.item())]).t(),dataset.vocab_size).to(device)
                    
                      print("TEXT "+ config.sample_method + " complete: " , txt)
                      file.write('Training Step: {}\n'.format(step))
                      file.write('--------------\n')
                      file.write('{}\n'.format(txt))
                      file.write('\n')
                

                
                if config.sample_method == "random":
                  if config.complete_sentence == False:
                    tau = config.temperature
                    softmax = torch.nn.Softmax(dim=1).to(device)
                    txt = np.random.choice(dataset._chars)
                    out = one_hot(torch.stack([torch.tensor(dataset._char_to_ix[txt])]).t(),dataset.vocab_size).to(device)
                    h0 = torch.zeros(config.lstm_num_layers, config.lstm_num_hidden ).to(device)
                    c0 = torch.zeros(config.lstm_num_layers, config.lstm_num_hidden ).to(device)
                    
                    for i in range(config.gen_seq_length):
                      out , (h0, c0) = model(out,h0,c0)
                      out = torch.argmax(softmax(tau*out))
                      txt = txt + dataset.convert_to_string([out.item()])
                      out = one_hot(torch.stack([torch.tensor(out.item())]).t(),dataset.vocab_size).to(device)
                    
                    print("TEXT "+ config.sample_method+": ", txt)
                    file.write('--------------\n')
                    file.write('Training Step: {}\n'.format(step))
                    file.write('--------------\n')
                    file.write('{}\n'.format(txt))
                    file.write('\n')
                  else:
                    tau = config.temperature
                    softmax = torch.nn.Softmax(dim=1).to(device)
                    txt = config.input_seq
                    # Generate some sentences by sampling from the model
                    h0 = torch.zeros(config.lstm_num_layers, config.lstm_num_hidden ).to(device)
                    c0 = torch.zeros(config.lstm_num_layers, config.lstm_num_hidden ).to(device)
                    for char in txt:
                      out = one_hot(torch.stack([torch.tensor(dataset._char_to_ix[char])]).t(),dataset.vocab_size).to(device)
                      _ , (h0, c0) = model(out,h0,c0)
                    out = txt[-1]  
                    out = one_hot(torch.stack([torch.tensor(dataset._char_to_ix[out])]).t(),dataset.vocab_size).to(device)

                    for i in range(config.gen_seq_length):
                      out , (h0, c0) = model(out,h0,c0)
                      out = torch.argmax(softmax(tau*out))
                      txt = txt + dataset.convert_to_string([out.item()])
                      out = one_hot(torch.stack([torch.tensor(out.item())]).t(),dataset.vocab_size).to(device)
                    
                    print("TEXT "+ config.sample_method + " complete: " , txt)
                    file.write('Training Step: {}\n'.format(step))
                    file.write('--------------\n')
                    file.write('{}\n'.format(txt))
                    file.write('\n')
            
            #if step % config.save_every == 0:
            #    torch.save(model.state_dict(), config.save_model)
            
            if step == config.train_steps-1:
                # If you receive a PyTorch data-loader error,
                # check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break
        if step == config.train_steps-1:
          break
        prev_steps = step+1
    file.close()    
    
    print('Done training.')
    dict= {"loss": losses, "acc": accuracies, "step": steps}
    
    if dict != None:
        return dict
###############################################################################
###############################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default = './assets/book_EN_grimms_fairy_tails.txt',
                        required=False,help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of an input sequence (training)')
    parser.add_argument('--gen_seq_length', type=int, default=30,
                        help='Length of generated sequence from the model')
    parser.add_argument('--lstm_num_hidden', type=int, default=128,
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=4e-3,
                        help='Learning rate')

    # It is not necessary to implement the following three params,
    # but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96,
                        help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000,
                        help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0,
                        help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=int(5e4),
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries",
                        help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5,
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100,
                        help='How often to sample from the model')
    parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
                        help="Device to run the model on.")

    #sample method
    parser.add_argument('--sample_method', type=str, default="greedy",
                        help="Choose sample method greedy or random")
    parser.add_argument('--complete_sentence', type=bool, default=False,
                        help="Complete a given sentence if True otherwise generate a sentence")
    parser.add_argument('--input_seq', type=str, default="Sleeping beauty is",
                        help="Input sequence to complete")
    parser.add_argument('--temperature', type=float, default=2.,
                        help="Set temperature for random sampling")
    #if you want plots
    parser.add_argument('--plots', type=str, default=True,
                        help="Plots")
    # If needed/wanted, feel free to add more arguments
    
    config = parser.parse_args()
    # Train the model
    if config.plots == False:
        train(config)
    else:
        import numpy as np
        import matplotlib.pyplot as plt

        parameters = "T=" + str(config.seq_length) + ' lr=' + str(config.learning_rate)  + ' batch_size=' + str(
                    config.batch_size)



        dict = train(config,dict)
        array_loss = np.array(dict["loss"])
        array_acc = np.array(dict["acc"])
        array_step = np.array(dict["step"])
        #print(array_loss)
        plt.figure()
        plt.title("loss curve " + "("+parameters+")")
        plt.plot(array_step ,array_loss,label="train")
        plt.ylabel('loss')
        plt.xlabel('step')
        plt.legend()
        plt.savefig('loss_pytorch.png', bbox_inches='tight')
        plt.figure()
        plt.title("accuracy curve" + "("+parameters+")")
        plt.plot(array_step,array_acc,label="train")
        plt.ylabel('accuracy')
        plt.xlabel('step')
        plt.legend()
        plt.savefig('acc_plot_T='+str(config.seq_length)+'.png', bbox_inches='tight')
        plt.show()