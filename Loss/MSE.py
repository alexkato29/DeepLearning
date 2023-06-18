class MSE():
    def __init__(self):
        pass


    def loss(self, y_pred, y_true):
        diff = y_pred - y_true
        m = y_true.shape[0]

        return (1/m) * (diff.T @ diff)
    

    def backward(self, y_pred, y_true):
        diff = y_pred - y_true
        return 2 * diff