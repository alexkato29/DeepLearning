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
    

    def backward(self, grad, Z):
        """
        Converts gradient to be pre-activation

        Parameters:
        - grad (vector): Vector of partial derivatives of cost wrt outputs of this layer POST activation

        Returns:
        - grad (vector): Vector of partial derivatives of cost wrt outputs of this layer PRE activation
        """

        # Piecewise. <= 0, then the deriv is 0. Otherwise, it's 1
        derivs = np.where(Z <= 0, 0, 1)
        grad = np.multiply(grad, derivs)
        
        return grad
    

    def __str__(self):
        return "ReLU"