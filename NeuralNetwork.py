import numpy as np
from Loss.MSE import MSE

class NeuralNetwork():
    """
    A simple feed-forward network architecture
    """
    def __init__(self, layers, loss, optimizer):
        # Initialize the layers and get a reverse layer list for backprop
        self.layers = layers
        self.n_layers = len(self.layers)

        # Initialize the loss function and optimizer
        self.criterion = loss.loss
        self.optimizer = optimizer

    
    def predict(self, X):
        """
        Predict a (set) of outputs.

        Parameters:
        - X (matrix): Input of the network with rows representing observations

        Outputs:
        - y (vector): Corresponding output, given the current set of weights
        """
        return self.forward(X)
    

    def fit(self, X, y):
        self.optimizer.fit(self, X, y)


    def forward(self, X):
        """
        Forward propagates the whole network to obtain outputs

        Parameters:
        - X (matrix): Input of the neural network with rows representing observations
        """
        for layer in self.layers:
            X = layer.forward(X)
        
        return(X)
    

    def backward(self, y):
        pass
    

    def get_layer(self, n):
        """
        Returns the nth layer within the network

        Parameters:
        - n (int): Index of the layer to be returned (starting at 0)
        """
        return self.layers[n]