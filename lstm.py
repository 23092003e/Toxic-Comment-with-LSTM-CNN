import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        """
        Initialize the LSTM parameters.
        input_size: Number of features in the input.
        hidden_size: Number of hidden units in the LSTM.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Forget gate parameters
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.bf = np.zeros((hidden_size, 1))

        # Input gate parameters
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.bi = np.zeros((hidden_size, 1))

        # Candidate cell state parameters
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.bc = np.zeros((hidden_size, 1))

        # Output gate parameters
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)
        self.bo = np.zeros((hidden_size, 1))
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        """Tanh activation function."""
        return np.tanh(x)
    
    def forward(self, x, h_prev, c_prev):
        """
        Perform a forward pass through the LSTM cell.
        x: Input at time step t (shape: input_size x 1).
        h_prev: Hidden state at time step t-1 (shape: hidden_size x 1).
        c_prev: Cell state at time step t-1 (shape: hidden_size x 1).
        """
        # Concatenate the previous hidden state and input
        combined = np.vstack((h_prev, x))
        
        # Forget gate
        f_t = self.sigmoid(np.dot(self.Wf, combined) + self.bf)
        
        # Input gate
        i_t = self.sigmoid(np.dot(self.Wi, combined) + self.bi)
        c_tilde_t = self.tanh(np.dot(self.Wc, combined) + self.bc)
        
        # Update cell state
        c_t = f_t * c_prev + i_t * c_tilde_t
        
        # Output gate
        o_t = self.sigmoid(np.dot(self.Wo, combined) + self.bo)
        h_t = o_t * self.tanh(c_t)
        
        return h_t, c_t
