"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample

        TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.

        Also, initialize gradients with zeros.
        """

        self.in_features = in_features
        self.out_features = out_features
        self.params = {}
        self.grads = {}

        self.params['weight'] = np.random.normal(loc=0, scale=0.0001, size=(self.out_features, self.in_features))
        # print(self.params['weight'].shape)
        self.params['bias'] = np.zeros(shape=(1, self.out_features))
        # self.params["dX"] = np.zeros_like()
        self.grads["weight"] = np.zeros_like(self.params['weight'])
        self.grads["bias"] = np.zeros_like(self.params["bias"])

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        self.params["X"] = x

        out = np.dot(self.params["X"], self.params['weight'].T) + self.params["bias"]

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        dx = np.dot(dout, self.params['weight'])
        # self.grads["dX"] = dx
        self.grads["weight"] = np.dot(dout.T, self.params["X"])
        self.grads["bias"] = np.sum(dout, axis=0)/ self.params["X"].shape[0]

        return dx


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        self.params = {}

        X_max = np.max(x, axis=1)
        numerator = np.exp(x - X_max.reshape(X_max.shape[0], 1))
        denominator = np.sum(numerator, axis=1)
        out = numerator / denominator.reshape(denominator.shape[0], 1)
        self.params["Y"] = out
        #print("Logits sum to:", np.sum(out, axis = 0))
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        dx = np.multiply(dout, self.params["Y"]) - np.einsum("in,in,ij->ij", dout, self.params["Y"], self.params["Y"])

        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        out = -(1 / x.shape[0]) * np.sum(np.multiply(np.log(x), y))

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #self.grads = {}

        dx = -(1 / x.shape[0]) * (y / x)
        return dx


class ELUModule(object):
    """
    ELU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        self.params = {}

        out = np.where(x > 0, x, np.exp(x) - 1)
        self.params["X"] = x
        # self.params["Y"] = out

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        dx = np.multiply(dout, np.where(self.params["X"] > 0, 1, np.exp(self.params["X"])))
        return dx



