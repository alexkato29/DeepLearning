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
    

    def backward(self, grad, Z):
        n = Z.shape[0]
        sig = self.forward(Z)

        # sig(Z) * (1 - sig(Z)) * grad
        sigprime = np.multiply(sig, (np.ones((n,1)) - sig))

        return np.multiply(grad, sigprime)
    
    
    def __str__(self):
        return "Sigmoid"
