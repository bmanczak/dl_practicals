"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn

class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """

    def __init__(self, n_channels=3, n_classes=10, act_fn=nn.ReLU):
        """
        Initializes ConvNet object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem


        TODO:
        Implement initialization of the network.
        """

        # The architecture
        super(ConvNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        self.conv0 = nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.PreAct1 = nn.Sequential(nn.BatchNorm2d(64),
                                     act_fn(),
                                     nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=False))

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)

        self.PreAct2a = nn.Sequential(nn.BatchNorm2d(128),
                                      act_fn(),
                                      nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=False))

        self.PreAct2b = nn.Sequential(nn.BatchNorm2d(128),
                                      act_fn(),
                                      nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=False))

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0)

        self.PreAct3a = nn.Sequential(nn.BatchNorm2d(256),
                                      act_fn(),
                                      nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False))

        self.PreAct3b = nn.Sequential(nn.BatchNorm2d(256),
                                      act_fn(),
                                      nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False))

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0)

        self.PreAct4a = nn.Sequential(nn.BatchNorm2d(512),
                                      act_fn(),
                                      nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False))

        self.PreAct4b = nn.Sequential(nn.BatchNorm2d(512),
                                      act_fn(),
                                      nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False))


        self.PreAct5a = nn.Sequential(nn.BatchNorm2d(512),
                                      act_fn(),
                                      nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False))

        self.PreAct5b = nn.Sequential(nn.BatchNorm2d(512),
                                      act_fn(),
                                      nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False))

        self.Classifier = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Flatten(),
                                        nn.Linear(512, n_classes))

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
        out = self.conv0(x)
        act1 = self.PreAct1(out)
        out = out + act1
        out = self.conv1(out)
        out = self.pool(out)

        act2a = self.PreAct2a(out)
        act2b = self.PreAct2b(out)
        out = out + act2a + act2b
        out = self.conv2(out)
        out = self.pool(out)

        act3a = self.PreAct3a(out)
        act3b = self.PreAct3b(out)
        out = out + act3a + act3b
        out = self.conv3(out)
        out = self.pool(out)

        act4a = self.PreAct4a(out)
        act4b = self.PreAct4b(out)
        out = out + act4a + act4b
        out = self.pool(out)

        act5a = self.PreAct5a(out)
        act5b = self.PreAct5b(out)
        out = self.pool(out)

        out = self.Classifier(out)

        return out