"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import sys
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    correct = np.sum(np.argmax(predictions,axis=1) == np.argmax(targets,axis=1))
    accuracy = correct / len(targets[:,0])
    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []
    #print(dnn_hidden_units)
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #take the other parameters from the flag
    lr = FLAGS.learning_rate 
    max_steps = FLAGS.max_steps
    eval_freq = FLAGS.eval_freq
    data_dir = FLAGS.data_dir
    batch_size = FLAGS.batch_size
    parameters = "hidden=" + str(FLAGS.dnn_hidden_units) + ' lr=' + str(FLAGS.learning_rate) + ' steps=' + str(FLAGS.max_steps) + ' batch_size=' + str(
                    FLAGS.batch_size) 
    #get data 
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    #print(cifar10)

    #for plotting
    loss_train =[]
    loss_test =[]
    acc_train = []
    acc_test = []
    steps = []
    #get test data
    n_inputs = 3*32*32
    n_classes = 10
    xtest, ytest = cifar10["test"].images, cifar10["test"].labels
    xtest = xtest.reshape(len(xtest[:,0,0,0]), n_inputs)
   
    
    #define model 
    model =  MLP(n_inputs, dnn_hidden_units,  n_classes)
    CrossEntropy = CrossEntropyModule() #loss function
    
    #training loop
    for step in range(max_steps):
        steps.append(step)

        #get batch - forward - loss 
        xtrain, ytrain = cifar10["train"].next_batch(batch_size)
        #we neet to reshape xtrain to be a flat vector
        xtrain = xtrain.reshape(FLAGS.batch_size, n_inputs)
        pred = model.forward(xtrain)
        loss = CrossEntropy.forward(pred, ytrain)
        
        loss_train.append(loss)
        acc_train.append(accuracy(pred, ytrain))
        #backward pass
        dout = CrossEntropy.backward(pred, ytrain)
        model.backward(dout) #doing this stores the grads in the linear layer dict self.grads

        #update weights and biases
        for layer in model.Linear_layers:
            layer.params["weight"] = layer.params["weight"] - lr * layer.grads["weight"]
            layer.params["bias"] = layer.params["bias"] - lr * layer.grads["bias"]
        
        #evaluation every eval_freq steps
        if (step % eval_freq == 0 ):
            test_pred = model.forward(xtest)
            test_loss = CrossEntropy.forward(test_pred, ytest)
            test_accuracy = accuracy(test_pred, ytest)
            loss_test.append(test_loss)
            acc_test.append(test_accuracy)
            sys.stdout = open(
                parameters +'_mlp_pytorch.csv', 'a')
            print("step: {}, loss: {:f}, acc: {:f}".format(step, test_loss, test_accuracy))
    
    #plotting
    plt.figure()
    plt.title("loss curve"+ "("+parameters+")")
    plt.plot(steps[::100],loss_train[::100],label="train")
    plt.plot(steps[::100],loss_test,label="test")
    plt.ylabel('loss')
    plt.xlabel('step') 
    plt.legend()
    plt.savefig('loss_numpy.png', bbox_inches='tight')
    plt.figure()
    plt.title("accuracy curve"+ "("+parameters+")")
    plt.plot(steps[::100],acc_train[::100],label="train")
    plt.plot(steps[::100],acc_test,label="test")
    plt.ylabel('accuracy')
    plt.xlabel('step')
    plt.legend()
    plt.savefig('acc_numpy.png', bbox_inches='tight')
    plt.show()
    ########################
    # END OF YOUR CODE    #
    #######################


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    #print(FLAGS)
    main()
