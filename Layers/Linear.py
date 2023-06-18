import numpy as np

class Linear():
    """
    Fully connected layer defined by an affine transformation W
    """

    def __init__(self, layer_size, activation):
        """
        Initialize a fully-connected layer

        Parameters:
        - layer_size (tuple (# inputs, # outputs)): Number of inputs and outputs of this layer
        - activation (class): Activation object to be used for this particular layer
        """

        # Number of inputs and outputs
        n_inputs = layer_size[0]
        n_outputs = layer_size[1]

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_weights = (n_inputs + 1) * n_outputs

        # Weight matrix
        # Weight vectors are the cols of the matrix, and the output will be of form XW for input X
        W = np.random.uniform(-1, 1, size=(self.n_inputs, self.n_outputs))
        zeros = np.zeros((1, self.n_outputs))
        self.W = np.vstack((zeros, W))

        # Activation
        self.activation = activation


    def forward(self, X):
        """
        Forward propagates the layer

        Parameters:
        - X ((n, m) matrix): Matrix of inputs with rows being observations. n observations, m features

        Returns:
        - A ((n, p) matrix): Output matrix of n rows and p cols, each representing the resulting output of that rows forward propagation
        """
        if X.shape[1] != (self.n_inputs):
            raise ValueError(f"Invalid layer input dimensions.")
        
        # Add a bias neuron
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))

        # Forward Propagate, store the results for use later
        self.Z = X @ self.W
        self.A = self.activation.forward(self.Z)

        return self.A


    def backward(self, X, y):
        pass
    

    def __str__(self):
        """
        Converts to a string of format "Fully Connected Layer: (n inputs, m outputs) -> Activation Function
        """
        return str.format("Fully Connected Layer: (%d inputs, %d outputs) -> %s" % (self.n_inputs, self.n_outputs, self.activation))