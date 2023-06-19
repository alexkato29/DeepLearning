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
        - X (matrix): Matrix to apply sigmoid to

        Returns:
        - A (matrix): Matrix with sigmoid funciton applied elementwise
        """
        t = np.exp((-1 * self.factor) * X)
        return np.reciprocal(1 + t)
    

    def backward(self, grad, Z, A):
        n = Z.shape[1]
        sig = A

        # sig(Z) * (1 - sig(Z)) * grad
        # sig(Z) = A
        sigprime = np.multiply(A, (np.ones((n,1)) - A))

        return np.multiply(grad, sigprime)
    
    
    def __str__(self):
        return "Sigmoid"
