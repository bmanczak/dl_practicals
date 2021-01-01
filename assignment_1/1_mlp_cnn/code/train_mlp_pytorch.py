"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math


# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100


# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
FLAGS = None

act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
}

"""
To replicate the 53% results:
python train_mlp_pytorch.py --optimizer "Adam" --learning_rate 0.001 --max_steps 3000 --dnn_hidden_units "256,128,64" --use_dropout True --use_batch_norm True --act_fn_name elu
"""

# Ways to initialize parametrs
def init_layers(net, act_fn_name):
    """Initializes the linear layers with Xavier initialization"""
    if act_fn_name == "tanh":
        for name, param in net.named_parameters():
            if len(param.shape) == 2: # let the rest be initialized by default
                nn.init.xavier_normal_(param)
    else: # ELU and ReLU
        for name, param in net.named_parameters():
            if len(param.shape) == 2:
                nn.init.kaiming_normal_(param)

def normal_init(net, std = 0.0001): # used to replicate numpy results
    for name, param in net.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        else:
            param.data.normal_(std = std)


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

    predictions = np.array(predictions)
    targets = np.array(targets)
    accuracy = (predictions.argmax(axis=1) == targets.argmax(axis=1)).sum() / predictions.shape[0]

    return accuracy


def test_model(net, dataset, loss_module, batch_size):
    """
    Test a model for accuracy and loss on a specified dataset. Extended upon the tutorial on Activation Functions.

    Inputs:
        net -MLP object
        dataset - dataset object (train or test)
        loss_module - module used to calculate the loss
        batch_size - size of fetchted batches
    """
    iters_needed = dataset._num_examples // batch_size  # iterations needed to check the whole dataset
    loss = []
    count = 0
    acc = []

    for i in range(iters_needed):
        imgs, labels = dataset.next_batch(batch_size)
        labels = torch.tensor(labels)
        imgs = imgs.reshape(imgs.shape[0], -1)
        with torch.no_grad():
            preds = net(imgs)
            count += labels.shape[0]
            acc.append(accuracy(preds, labels))
            loss.append(loss_module(preds, torch.max(labels, 1)[1]).item())

    acc = np.mean(acc)
    loss = np.mean(loss)

    return acc, loss


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


    # Initialize the network and loss module
    if FLAGS.use_dropout == True:
        if FLAGS.use_batch_norm == True:
            the_mlp = MLP(3072, dnn_hidden_units, 10, act_fn = act_fn_by_name[FLAGS.act_fn_name](), use_dropout = True, use_batch_norm = True)
        else:
            the_mlp = MLP(3072, dnn_hidden_units, 10, act_fn = act_fn_by_name[FLAGS.act_fn_name](), use_dropout=True, use_batch_norm=False)
    else:
        if FLAGS.use_batch_norm == True:
            the_mlp = MLP(3072, dnn_hidden_units, 10,act_fn = act_fn_by_name[FLAGS.act_fn_name](), use_dropout = False, use_batch_norm = True)
        else:
            if FLAGS.use_batch_norm == False:
                the_mlp = MLP(3072, dnn_hidden_units, 10, act_fn = act_fn_by_name[FLAGS.act_fn_name](),use_dropout=False, use_batch_norm=False)
                normal_init(the_mlp, std = 0.0001) # replicate NumPy results

    if FLAGS.manual_init == True:
        init_layers(the_mlp, FLAGS.act_fn_name)

    if FLAGS.optimizer == "Adam":
        optimizer = optim.Adam(the_mlp.parameters(), lr=FLAGS.learning_rate, weight_decay = 1e-3)
    elif FLAGS.optimizer == "SGD":
        optimizer = optim.SGD(the_mlp.parameters(), lr=FLAGS.learning_rate)  # Default parameters, feel free to change
    else:
        raise NotImplementedError("Optimizer not supported!")


    loss_module = nn.CrossEntropyLoss()

    # Load the data generator
    cifar10 = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)
    step = 0  # counter for steps

    # Train/test split
    trainset = cifar10['train']
    testset = cifar10['test']
    accuracies_test = []
    accuracies_train = []
    losses_train = []
    losses_test = []

    while step <= FLAGS.max_steps:
        imgs, labels = trainset.next_batch(FLAGS.batch_size)
        labels = torch.tensor(labels)
        # imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        imgs = imgs.reshape(imgs.shape[0], -1)
        preds = the_mlp(imgs)

        loss = loss_module(preds, torch.max(labels, 1)[1])

        # Backward step
        loss.backward()
        optimizer.step()

        if step % FLAGS.eval_freq == FLAGS.eval_freq - 1:  #
            print("[INFO]: Evaluation at step {} ...".format(step))
            acc_at_step_test, loss_at_step_test = test_model(the_mlp, testset, loss_module, FLAGS.batch_size)
            acc_at_step_train, loss_at_step_train = test_model(the_mlp, trainset, loss_module, FLAGS.batch_size)
            accuracies_test.append(round(acc_at_step_test * 100, 4))
            accuracies_train.append(round(acc_at_step_train * 100, 4))
            losses_test.append(loss_at_step_test)
            losses_train.append(loss_at_step_train)
            print("[INFO]: Train acc {}%, Test acc {}%".format(round(acc_at_step_train * 100, 4), round(acc_at_step_test * 100, 4)))
            print("[INFO]: Train loss {}, Test loss {}".format(loss_at_step_train, loss_at_step_test))

        step += 1

    return the_mlp, [accuracies_test, accuracies_train, losses_test, losses_train]


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
    _, stats = train()
    # Log the plot
    plot_accuracy_loss_curves(stats, eval_freq= FLAGS.eval_freq)

def plot_accuracy_loss_curves(stats, eval_freq, save_dir = "./figs"):
    """
    Plots the accuracy and loss curve on one, double y-axis plot. Saves the plot in a given directory.

    """
    accuracies_test, accuracies_train, losses_test, losses_train = stats
    step = np.arange(start=eval_freq, stop=len(accuracies_test) * eval_freq + 1, step=eval_freq) # shared x axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(step, accuracies_test, label="Test Accuracy", color="blue")
    ax.plot(step, accuracies_train, label="Train Accuracy", color="red")

    ax.set_xlabel("Step", size=30)
    ax.set_ylabel("Accuracy[%]", size=25)
    plt.xticks(size=20)
    plt.yticks(size=20)

    ax2 = ax.twinx()
    ax2.plot(step, losses_test, label="Test Loss", color="black")
    ax2.plot(step, losses_train, label="Train Loss", color="orange")
    ax2.set_ylabel("Loss", size=25)
    plt.yticks(size=20)

    fig.legend(prop={'size': 15}, bbox_to_anchor=(0.0, 1))
    fig.suptitle("Accuracy and loss curves for training and validation phase", size=20)
    plt.tight_layout()

    if os.path.exists(FLAGS.save_dir):
        fig.savefig(os.path.join(FLAGS.save_dir, "loss_and_accuracy_curve_pytorch_mlp.jpg"), bbox_inches='tight')
        print("[INFO]: Loss and accuracy curve saved in:", FLAGS.save_dir)
    else:
        fig.savefig("loss_and_accuracy_curve_pytorch_mlp.jpg", bbox_inches='tight')
        print("[INFO]: Loss and accuracy curve saved in current working directory:", os.getcwd())

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
    parser.add_argument('--optimizer', type=str, default="SGD",
                        help='What optimizer to use')
    parser.add_argument('--manual_init', type=bool, default=False,
                        help='Whether to perform manual, Kaiming init for Elu/ReLU or Xavier.')
    parser.add_argument('--use_dropout', type=bool, default=False,
                        help='Whether to use dropout before the last linear layer')
    parser.add_argument('--use_batch_norm', type=bool, default=False,
                        help='Whether to use batch normalization after each linear layer')
    parser.add_argument('--act_fn_name', type=str, default="elu",
                        help='What activation function to use')
    parser.add_argument('--save_dir', type=str, default="./figs",
                        help='Directory where to store the plot. If does not exist,'
                             'saves to current directory.')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()