import numpy as np

class Linear():
    def __init__(self, layer_size, activation, dropout=0.9):
        """
        Initialize a fully-connected layer defined by an affine transformation W

        Parameters:
        - layer_size (tuple (# inputs, # outputs)): Number of inputs and outputs of this layer
        - activation (class): Activation object to be used for this particular layer
        - dropout (float): Dropoout proportion for this layer
        """

        # Number of inputs and outputs
        n_inputs = layer_size[0]
        n_outputs = layer_size[1]

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_weights = (n_inputs + 1) * n_outputs

        # Weight matrix w/ He initialization
        # Weight vectors are the cols of the matrix, and the output will be of form XW for input X
        self.W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2./n_inputs)

        # Bias vector. Row vector since it will be added to observations, which are rows
        self.b = np.zeros((1, self.n_outputs))

        # Activation
        self.activation = activation

        # Dropout
        self.dropout = dropout


    def forward(self, X):
        """
        Forward propagates the layer

        Parameters:
        - X ((n, m) matrix): Matrix of inputs with rows being observations. n observations, m features

        Returns:
        - A ((n, p) matrix): Output matrix of n rows and p cols, each representing the resulting output of that rows forward propagation
        """
        if X.shape[1] != (self.n_inputs):
            raise ValueError(f"Invalid layer input dimensions.")
        
        # Store X for backprop
        self.X = X

        # Forward Propagate, store the results for use later
        self.Z = (X @ self.W) + self.b
        self.A = self.activation.forward(self.Z)

        # Apply dropout
        d = np.random.binomial(1, self.dropout, self.n_outputs).T
        self.A = self.A * d / self.dropout

        return self.A


    def backward(self, grad):
        grad_bwa = []  # Three gradients: wrt b, W, A, respectively

        # Get the gradient PRE activation, dC/dZ
        # This also happens to be the gradient wrt the bias 
        grad = self.activation.backward(grad, self.Z, self.A)
        grad_b = grad
        grad_bwa.append(grad_b)

        # Get gradient of C wrt W
        grad_W = self.X.T @ grad
        grad_bwa.append(grad_W)

        # Get gradient of C wrt A
        grad_A = np.sum((self.W @ grad.T), axis=0)
        grad_bwa.append(grad_A) 

        return grad_bwa
    

    def update(self, db, dW, lr):
        """
        Update the weights and biases of the layer.

        Parameters:
        - db (vector): Gradient vector of outputs wrt biases
        - dW (matrix): Jacobian matrix of outputs wrt weights
        """
        self.b = self.b - (lr * db)
        self.W = self.W - (lr * dW)
    

    def __str__(self):
        """
        Converts to a string of format "Fully Connected Layer: (n inputs, m outputs) -> Activation Function
        """
        return str.format("Fully Connected Layer: (%d inputs, %d outputs) -> %s" % (self.n_inputs, self.n_outputs, self.activation))