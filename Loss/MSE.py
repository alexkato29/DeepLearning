import numpy as np

class MSE():
    def __init__(self):
        """
        Initializes a mean squared error loss
        """
        pass


    def loss(self, y_true, y_pred):
        """
        Computes the mean squared error

        Parameters:
        - y_true (vector): True values for all observations
        - y_pred (vector): Predicted values for all observations

        Returns:
        - mse (float): Mean squared error of the predictions
        """
        resid = y_true - y_pred
        sq_resid = np.square(resid)
        ssr = np.sum(sq_resid)
        size = y_true.shape[0] * y_true.shape[1]
        mse = (1/size) * ssr

        return mse
    

    def backward(self, y_true, y_pred):
        """
        Computes the gradient of the loss function w.r.t the model outputs

        Parameters:
        - y_true (vector): True values for all observations
        - y_pred (vector): Predicted values for all observations

        Returns:
        - grad (vector): Gradient of the cost function w.r.t all outputs
        """
        return -2 * (y_true - y_pred)