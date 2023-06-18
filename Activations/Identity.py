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
    

    def backward(self):
        pass
    
    
    def __str__(self):
        return "Identity"
