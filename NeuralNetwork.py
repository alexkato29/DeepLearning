import numpy as np
from Loss.MSE import MSE

class NeuralNetwork():
    """
    A simple feed-forward network architecture
    """
    def __init__(self, layers, criterion):
        # Initialize the layers and get a reverse layer list for backprop
        self.layers = layers
        self.n_layers = len(self.layers)

        # Initialize the loss function
        self.criterion = criterion
    
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
    

    def fit(self, X, y, batch_size=30, epochs=1000, lr=0.05, verbose=False, show_grad=False):
        """
        Fit the network on the given data. Uses BatchGD only for now, will need to change this in a future version.

        Parameters:
        - X ((n, m) matrix): Observations 
        - y ((n, 1) matrix): Observed outputs
        - epochs (int): # of times to run through the whole training set
        - verbose (boolean): If true, will print information of every 10th epoch
        """
        n = X.shape[0]

        for epoch in range(epochs + 1):

            # Randomly shuffle the indices
            rand_indices = np.random.permutation(X.shape[0])
            X_shuffled = X[rand_indices]
            y_shuffled = y[rand_indices]
            
            # Create the batches
            X_batches = [X_shuffled[i:i+batch_size] for i in range(0, n, batch_size)]
            y_batches = [y_shuffled[i:i+batch_size] for i in range(0, n, batch_size)]

            # Retrain for each batch
            for i in range(0, len(X_batches)):
                # Reset the gradient list
                self.grad_W = [np.zeros(layer.W.shape) for layer in reversed(self.layers)]
                self.grad_b = [np.zeros(layer.b.shape) for layer in reversed(self.layers)]

                for j in range(0, len(X_batches[i])):
                    # Get the jth observation in the mini batch
                    jth_obs = np.array([X_batches[i][j]])
                    jth_target = np.array([y_batches[i][j]])
                
                    # Update layer outputs
                    y_pred = self.forward(jth_obs)

                    # Update gradients
                    self.backward(jth_target, y_pred)

                for index, layer in enumerate(reversed(self.layers)):
                    # Update the layers
                    layer.update(self.grad_b[index], self.grad_W[index], lr)

                if show_grad:
                    print("Bias Gradients: ")
                    print(self.grad_b)
                    print("Weight Jacobians: ")
                    print(self.grad_W)

            # Print information if verbose
            if verbose and epoch % 20 == 0:
                y_pred = self.predict(X_shuffled)
                loss = self.criterion.loss(y_shuffled, y_pred)
                print(f"Epoch [{epoch}/{epochs}], Loss: {loss:.4f}")
                 
    

    def backward(self, y_actual, y_pred):
        """
        Backward propogates through the network to compute the partial derivatives wrt each weight/bias.

        Parameters:
        - y_pred (vector): outputs of the network
        - y_actual (vector): actual values of the target
        """

        # You must compute gradients for each observation, then average them

        # Compute the start gradient from the loss function
        grad = self.criterion.backward(y_actual, y_pred)

        for index, layer in enumerate(reversed(self.layers)):
            # Compute the gradients 
            grads = layer.backward(grad)

            # Update them here
            self.grad_b[index] += grads[0]
            self.grad_W[index] += grads[1]
            grad = grads[2]
        


    def get_layer(self, n):
        """
        Returns the nth layer within the network

        Parameters:
        - n (int): Index of the layer to be returned (starting at 0)
        """
        return self.layers[n]
    

    def show(self, detailed=False):
        for layer in self.layers:
            print(layer)
            if detailed:
                print("Weights (column corresponds to output neuron): ")
                print(layer.W)
                print("Biases: ")
                print(layer.b)