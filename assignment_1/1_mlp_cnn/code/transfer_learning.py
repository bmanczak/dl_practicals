"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import matplotlib
import matplotlib.pyplot as plt
import math


## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision import models


# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
#DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
DATA_DIR_DEFAULT = './cifar10'

FLAGS = None


def test_model(net, data_loader, loss_module, device):
    """
    Test a model on a specified dataset.

    Inputs:
        net - Trained model of type BaseNetwork
        data_loader - DataLoader object of the dataset to test on (validation or test)
    """
    net.eval()
    true_preds, count, loss, count_batches = 0, 0, 0, 0

    for imgs, labels in data_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            net_out = net(imgs)
            preds = net_out.argmax(dim=-1)
            true_preds += (preds == labels).sum().item()
            count += labels.shape[0]
            count_batches += 1
            loss += loss_module(net_out, labels)  # .item()

    test_acc = true_preds / count
    loss = loss / count_batches
    return test_acc, loss


def set_parameter_requires_grad(model, feature_extracting, trainable_name):
    """Sets all parameters to not trainable except for the ones specified in trainable_name"""
    if feature_extracting:
        for name, param in model.named_parameters():
            if trainable_name in name:
                param.requires_grad = True
            else:
                param.requires_grad = False



def train():
    """
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                      transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
    test_transform = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ])

    trainset = torchvision.datasets.CIFAR10(root=FLAGS.data_dir, train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 32,
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root=FLAGS.data_dir, train=False,
                                           download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=4)
    # Initialize the network and loss module
    net = models.resnet18(pretrained=True) # load model

    trainable_name = "layer4" # unfreeze the 4-th block
    set_parameter_requires_grad(net, True, trainable_name)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 128) # reinstatement sets train to True for the final linear layer
    net.d = nn.Dropout2d()
    net.fc2 = nn.Linear(128, 10)

    # Fetching the device that will be used throughout this notebook
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print("[INFO]: Using device", device)

    if torch.cuda.is_available():
      net.cuda()

    if OPTIMIZER_DEFAULT=="ADAM":
      optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE_DEFAULT)
    loss_module = nn.CrossEntropyLoss()


    step = 0  # counter for steps


    accuracies_test = []
    accuracies_train = []
    losses_train = []
    losses_test = []


    while step <= MAX_STEPS_DEFAULT:
      for imgs, labels in trainloader:
        if step >= MAX_STEPS_DEFAULT:
          break
        imgs, labels = imgs.to(device), labels.to(device) # To GPU
        optimizer.zero_grad()
        preds = net(imgs)

        loss = loss_module(preds,labels)

        # Backward step
        loss.backward()
        optimizer.step()

        if step % EVAL_FREQ_DEFAULT == EVAL_FREQ_DEFAULT - 1:  #
          print("[INFO]: Evaluation at step {} ...".format(step))
          acc_at_step_test, loss_at_step_test = test_model(net, testloader, loss_module, device)
          acc_at_step_train, loss_at_step_train = test_model(net, trainloader, loss_module, device)
          accuracies_test.append(round(acc_at_step_test * 100, 4))
          accuracies_train.append(round(acc_at_step_train * 100, 4))
          losses_test.append(loss_at_step_test)
          losses_train.append(loss_at_step_train)
          print("[INFO]: Train acc {}%, Test acc {}%".format(round(acc_at_step_train * 100, 4), round(acc_at_step_test * 100, 4)))
          print("[INFO]: Train loss {}, Test loss {}".format(loss_at_step_train, loss_at_step_test))
        step += 1

      if step >= MAX_STEPS_DEFAULT:
          break

    return net, [accuracies_test, accuracies_train, losses_test, losses_train]

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def plot_accuracy_loss_curves(stats, eval_freq, save_dir):
    """
    Plots the accuracy and loss curve on one, double y-axis plot. Saves the plot in a given directory.

    Args:

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
        fig.savefig(os.path.join(FLAGS.save_dir, "loss_and_accuracy_curve_transfer_learning_cnn.jpg"), bbox_inches='tight')
        print("[INFO]: Loss and accuracy curve saved in:", FLAGS.save_dir)
    else:
        fig.savefig("loss_and_accuracy_curve_transfer_learning_cnn.jpg", bbox_inches='tight')
        print("[INFO]: Loss and accuracy curve saved in current working directory",os.getcwd())

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
    plot_accuracy_loss_curves(stats, eval_freq=FLAGS.eval_freq, save_dir=FLAGS.save_dir)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--save_dir', type=str, default="./figs",
                        help='Directory where to store the plot')
    FLAGS, unparsed = parser.parse_known_args()

    main()
