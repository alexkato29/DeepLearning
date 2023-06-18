import numpy as np

class Identity():
    """
    Identity activation function
    """

    def __init__(self):
        """
        Initialize an Identity activation function for the preceeding layer.
        """
        pass


    def forward(self, X):
        """
        Identity of X matrix

        Parameters:
        - X (matrix)

        Returns:
        - X (matrix)
        """
        return X
    

    def backward(self, grad, Z):
        """
        Converts gradient to be pre-activation

        Parameters:
        - grad (vector): Vector of partial derivatives of cost wrt outputs of this layer POST activation

        Returns:
        - grad (vector): Vector of partial derivatives of cost wrt outputs of this layer PRE activation
        """

        # No activation, so the gradient already is the "pre" activation derivative
        return grad
    
    
    def __str__(self):
        return "Identity"
