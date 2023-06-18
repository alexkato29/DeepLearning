class MSE():
    def __init__(self):
        pass


    def loss(self, y_true, y_pred):
        diff = y_true - y_pred
        m = y_true.shape[0]

        return (1/m) * (diff.T @ diff)[0,0]
    

    def backward(self, y_true, y_pred):
        m = y_true.shape[0]
        diff = y_true - y_pred

        return (-2/m) * diff