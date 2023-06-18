import numpy as np
from Loss.MSE import MSE

class NeuralNetwork():
    """
    A simple feed-forward network architecture
    """
    def __init__(self, layers, criterion, optimizer):
        # Initialize the layers and get a reverse layer list for backprop
        self.layers = layers
        self.n_layers = len(self.layers)

        # Initialize the loss function and optimizer
        self.criterion = criterion
        self.optimizer = optimizer

    
    def predict(self, X):
        """
        Predict a (set) of outputs.

        Parameters:
        - X (matrix): Input of the network with rows representing observations

        Outputs:
        - y (vector): Corresponding output, given the current set of weights
        """
        return self.forward(X)


    def forward(self, X):
        """
        Forward propagates the whole network to obtain outputs

        Parameters:
        - X (matrix): Input of the neural network with rows representing observations
        """
        for layer in self.layers:
            X = layer.forward(X)
        
        return(X)
    

    def fit(self, X, y, batch_size=30, epochs=1000, verbose=False):
        """
        Fit the network on the given data. Uses BatchGD only for now, will need to change this in a future version.

        Parameters:
        - X ((n, m) matrix): Observations 
        - y ((n, 1) matrix): Observed outputs
        - epochs (int): # of times to run through the whole training set
        - verbose (boolean): If true, will print information of every 10th epoch
        """
        n = X.shape[0]

        for epoch in range(epochs):

            # Randomly shuffle the indices
            rand_indices = np.random.permutation(X.shape[0])
            X_shuffled = X[rand_indices]
            y_shuffled = y[rand_indices]
            
            # Create the batches
            X_batches = [X_shuffled[i:i+batch_size] for i in range(0, n, batch_size)]
            y_batches = [y_shuffled[i:i+batch_size] for i in range(0, n, batch_size)]

            # Retrain for each batch
            for i in range(0, len(X_batches)):
                
                # Update layer outputs
                y_pred = self.forward(X_batches[i])

                # Update gradients
                self.backward(y_batches[i], y_pred)

                # Send gradients to the optimizer for usage
                # self.optimizer.step(self.gradients)

            # Print information if verbose
            if verbose and epoch % 10 == 0:
                y_pred = self.predict(X_shuffled)
                loss = self.criterion.loss(y_shuffled, y_pred)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}")
                 
    

    def backward(self, y_actual, y_pred):
        """
        Backward propogates through the network to compute the partial derivatives wrt each weight/bias.

        Parameters:
        - y_pred (vector): outputs of the network
        - y_actual (vector): actual values of the target
        """

        # BE MINDFUL THAT YPRED IS POTENTIALLY  A MATRIX
        # You must compute gradients for each observation, then average them

        # Reset the gradient list
        self.gradients = []

        # Compute the start gradient from the loss function
        grad = self.criterion.backward(y_actual, y_pred)

        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            self.gradients.append(grad)
        


    def get_layer(self, n):
        """
        Returns the nth layer within the network

        Parameters:
        - n (int): Index of the layer to be returned (starting at 0)
        """
        return self.layers[n]