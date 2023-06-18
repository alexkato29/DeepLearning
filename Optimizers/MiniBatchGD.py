import numpy as np

class MiniBatchGD():
    """ 
    MiniBatch Gradient Descent Optimizer
    """
    def __init__(self, batch_size=30, epochs=100, lr=0.05, schedule="adaptive"):
        """
        Initialize an instance of Animal.

        Parameters:
            name (str): The name of the animal.
        """

        # Set params
        self.batch_size = batch_size
        self.lr = lr
        self.schedule = schedule
        self.gradients = []


    def reset_gradients(self):
        self.gradients = []


    def fit(self, nn, X, y):
        for epoch in range(self.epochs):
            np.random.shuffle(X)
            batches = [X[i:i+self.batch_size] for i in range(0, X.shape[0], self.batch_size)]
            for batch in batches:
                # Update gradients
                # Take a step in direction for each gradient
                pass                


    def step(self, X, y, epochs, verbose, batch_size, eta):
        n, p = X.shape  # n observations, p features
        data = np.hstack((y.copy(), X.copy()))

        for epoch in range(epochs):
            np.random.shuffle(data)
            mini_batches = [data[k:k+batch_size,] for k in range (0, n, batch_size)]

            for batch in mini_batches:
                X_i = batch[:,1:]
                y_i = batch[:,0]

                for i in range(0, self.n_layers - 1):
                    layer = self.rev_layers[i]
                    print(layer)
