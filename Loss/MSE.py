import numpy as np

class MSE():
    def __init__(self):
        pass


    def loss(self, y_true, y_pred):
        resid = y_true - y_pred
        sq_resid = np.square(resid)
        ssr = np.sum(sq_resid)
        size = y_true.shape[0] * y_true.shape[1]

        return (1/size) * ssr
    

    def backward(self, y_true, y_pred):
        return -2 * (y_true - y_pred)