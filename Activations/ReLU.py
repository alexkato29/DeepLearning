import numpy as np

class Relu():
    """
    ReLU activation function
    """
    
    def __init__(self):
        """
        Initialize a ReLU activation for the preceeding layer.
        """
        pass
    
    
    def forward(self, X):
        """
        ReLU of X matrix

        Parameters:
        - X (matrix): Matrix to apply ReLU to

        Returns:
        - A (matrix): Matrix with ReLU function applied elementwise
        """
        return np.maximum(0, X)
    

    def __str__(self):
        return "ReLU"