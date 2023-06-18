import numpy as np

class Sigmoid():
    """
    Sigmoid activation function
    """

    def __init__(self, factor=1):
        """
        Initialize a sigmoid activation for the preceeding layer.

        Parameters:
        - factor (int): Factor by which to scale the exponent feature of the sigmoid. Stretches or squishes the activation curve.
        """
        self.factor = factor


    def forward(self, X):
        """
        Sigmoid of X matrix

        Parameters:
        - x ((n, p) matrix): Matrix to apply sigmoid to
        """
        t = np.exp((-1 * self.factor) * X)
        return np.reciprocal(1 + t)
    

    def backward(self):
        pass
    
    
    def __str__(self):
        return "Sigmoid"
