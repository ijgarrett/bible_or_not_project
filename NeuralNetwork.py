import numpy as np

# ------------------------------
# LAYERS
# ------------------------------

# Dense Layer, fully connected
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons, l2_lambda=0.0):
        # Initialize weights and biases
        # returns a matrix of size (n_inputs, n_neurons) with values from the normal distribution of mean 0 std 1
        # multiplying by 0.01 scales values down to smaller, random weights
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

        # creates the bias vector, initially all zeros. shape (1, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # L2 regularization lambda
        # Loss = Loss + (lambda / 2) * sum(Wij)^2
        # derivative w.r.t. weights gets you lambda * W in backward pass
        self.l2_lambda = l2_lambda

    # Forward pass
    def forward(self, inputs):
        # Remember inputs
        # size of inputs: (batch_size, n_inputs)
        self.inputs = inputs

        # calculate output values from inputs, weights, and biases
        # size: (batch_size, n_neurons)
        self.output = np.dot(self.inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # dvalues = dL/dZ, or the gradient of the loss with respect to the output of this layer. size: (batch_size, n_neurons)
        # Gradients on parameters

        # gradient of loss with respect to the weights for the ith input feature to the jth neuron
        self.dweights = np.dot(self.inputs.T, dvalues) + self.l2_lambda * self.weights  # size: (n_inputs, n_neurons)

        # gradient of loss with respect to the biases.
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)  # size: (1, n_neurons)
        
        # gradient of loss with respect to input values
        self.dinputs = np.dot(dvalues, self.weights.T)  # size: (batch_size, n_inputs)

# RELU ACTIVATION: FORWARD PASS
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        # Remember inputs
        # inputs are the z pre activation values. size (batch_size, n_neurons)
        self.inputs = inputs

        # Calculate output values from inputs. ReLU function formula: max(0, x)
        self.output = np.maximum(0, inputs)  # size (batch_size, n_neurons)

    # Backward pass
    def backward(self, dvalues):
        # copy the dL/dZ matrix
        self.dinputs = dvalues.copy()
        
        # wherever x <= 0, change corresponding indices in dL/dZ to 0
        # Remember, self.inputs are the outputs from the hidden layer
        self.dinputs[self.inputs <= 0] = 0
        
        # dvalues, dinputs, and inputs all have size (batch_size, n_neurons)

# SOFTMAX ACTIVATION: FORWARD PASS
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
        # inputs, exp_values, probabilities, and self.output are size (batch_size, n_neurons)
        # Get innormalized probabilities
        # subract the max value of each row of inputs (per sample) to prevent overflow when exponentiating big numbers
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        # Normalize them for each sample
        # divide each row by its exponentiated row sum
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # Backward pass (won't use)
    def backward(self, dvalues):
        # dvalues = dL/dZ, where Z is softmax output. shape (batch_size, n_neurons)
        # create empty array the same shape as dvalues
        self.dinputs = np.empty_like(dvalues)

        # Iterate over each sample in the batch
        # zip goes through one (single_output, single_dvalues) at a time
        # enumerate assigns index to an iterable
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output vector for current sample as column vector 
            # -1 tells python to figure it out. 1 means for sure 1 column
            single_output = single_output.reshape(-1, 1)

            # Compute Jacobian matrix of the softmax function:
            # J = diag(softmax) - softmax * softmax.T
            # diagflat creates a 2d diagonal matrix from a 1d (flattened) array
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient: dL/dinputs = J * dL/doutputs
            # insert it into the row for each sample as the for loop progresses
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

# ------------------------------
# LOSS CLASS
# ------------------------------

# Common loss class
class Loss:
    # Calculates the data and regularization losses given model output and ground truth values
    def calculate(self, output, y):
        # Calculate sample losses
        # forward is assumed to be implemented in subclasses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0 and log(0)
        # Clip both sides to not drag the mean towards the left or right
        # if there is a 0, it's clipped to 0.0000001. if there's a 1, it's clipped to 0.9999999
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values if categorical labels: 1d array. Ex dimension would be (4,), so len is 1
        if len(y_true.shape) == 1:
            # range(samples) creates the row indices for each sample
            # y_true contains the column indices for the correct class of each sample
            # for each sample i, it picks the predicted probability from y_pred_clipped at row i and column y_true[i]
            correct_confidences = y_pred_clipped[range(samples), y_true]
            
        # Mask values - only for one-hot encoded labels: use one array to select certain elements from another array
        elif len(y_true.shape) == 2:
            # * does element wise multiplication
            # axis = 1 computes a row wise sum
            # result is a 1D array (no keepdim) of size (batch_size, )
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Return the negative log likelihoods
        return -np.log(correct_confidences)   

    # Backward pass (won't use)
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in each sample
        labels = len(dvalues[0])
        # if the y_true are categorical labels instead of one-hot, switch it
        if len(y_true.shape) == 1:
            # create an identity matrix with length number of labels
            # selects rows from identity matrix corresponding with the indices in y_true
            y_true = np.eye(labels)[y_true]

        # calculate the gradient
        self.dinputs = -y_true / dvalues
        # normalize the gradient to average loss
        self.dinputs = self.dinputs / samples

# ------------------------------
# COMBINED SOFTMAX ACTIVATION AND CATEGORICAL CROSS ENTROPY FOR LAST LAYER: FORWARD AND BACKWARD PASS
# ------------------------------

class Activation_Softmax_Loss_CategoricalCrossEntropy:
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # inputs is size (batch_size, n_classes)
        # y is size (batch_size, n_classes) if one hot encoded, otherwise (batch_size,)
        # Output layer's activation function
        # Applies softmax activation to the input logits
        # Stored in self.activation.outputs
        self.activation.forward(inputs)
        
        # Set the output
        self.output = self.activation.output # (batch_size, n_classes)
        
        # Calculate and return loss value
        # Calls the loss function on each sample, then returns the average loss
        return self.loss.calculate(self.output, y_true) # scalar

    # Backward pass
    def backward(self, dvalues, y_true):
        # dvalues has shape (batch_size, n_classes)
        # y_true has shape (batch_size, n_classes) if one hot encoded otherwise (batch_size,)
        
        # Number of samples
        samples = len(dvalues)
        
        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            # argmax returns the index of the max value along a given axis
            # we use axis = 1 to look across each row
            y_true = np.argmax(y_true, axis=1)
            
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        
        # Calculuate the gradient: formula is dL/dX = Z - Y = Prediction - Ground Truth
        # Y would be one hot encoded, so only 1 index per row in dinputs would be subtracted by 1, the rest by 0
        # but since y_true just contains class indices, subtract 1 from the corresponding indices in each row of dinputs
        self.dinputs[range(samples), y_true] -= 1
        
        # Normalize gradient
        self.dinputs /= samples

# ------------------------------
# ADAM OPTIMIZER
# ------------------------------

# Adam optimizer
class Optimizer_Adam:
    # Initialize optizer - set settings
    def __init__(self, learning_rate = 0.001, decay = 0., epsilon = 1e-7, beta_1 = 0.9, beta_2 = 0.999):
        self.current_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.decay = decay # default 0
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates   
    def pre_update_params(self):
        # check if there is a decay rate
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)

    # Update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights) # (n_inputs, n_neurons)
            layer.weight_cache = np.zeros_like(layer.weights) # (n_inputs, n_neurons)
            layer.bias_momentums = np.zeros_like(layer.biases) # (1, n_neurons)
            layer.bias_cache = np.zeros_like(layer.biases) # (1, n_neurons)

        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Get corrected momentum
        # self.iteration is 0 at first pass and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * (layer.dweights ** 2)
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * (layer.dbiases ** 2)

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected + self.epsilon))
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected + self.epsilon))

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Layer_Dropout:

    def __init__(self, rate):
        # rate = prob of dropping a neuron output
        self.rate = rate

    def forward(self, inputs):
        self.inputs = inputs

        # np.random(num_trials, prob success, size)
        self.binary_mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape) / (1 - self.rate)

        self.output = inputs * self.binary_mask
    
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


# ------------------------------
# NEURAL NETWORK CLASS
# ------------------------------
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X, y=None):
        for layer in self.layers:
            # Combined softmax-loss layer needs y only if calculating loss
            if isinstance(layer, Activation_Softmax_Loss_CategoricalCrossEntropy):
                if y is not None:
                    output = layer.forward(X, y)  # during training/loss computation
                else:
                    # just do softmax activation for prediction
                    layer.activation.forward(X)
                    output = layer.activation.output
            else:
                # For all other layers like Layer_Dense or Activation_ReLU
                output = layer.forward(X) or layer.output # layer.output for safety
            X = output

        # final output of the network, either raw softmax probs during prediction or loss for training if called with y
        return X


    # Backward pass through all layers
    # y contains true labels
    def backward(self, y):
        # Start with last layer
        last_layer = self.layers[-1]
        if isinstance(last_layer, Activation_Softmax_Loss_CategoricalCrossEntropy):
            last_layer.backward(last_layer.output, y)
            dvalues = last_layer.dinputs
        else:
            raise Exception("Last layer must be Activation_Softmax_Loss_CategoricalCrossEntropy")

        # Backpropagate through hidden layers
        # loop through layers in reverse, excluding the last layer
        for layer in reversed(self.layers[:-1]):
            layer.backward(dvalues)
            dvalues = layer.dinputs

    # Training
    # X, y: the training data and labels
    # epochs: # passes over entire dataset
    # batch_size: size of mini batches
    # optimizer: instance of optimizer
    # verbose: whether to print progress
    # X_val, y_val: optional validation data
    def train(self, X, y, epochs=1000, batch_size=None, optimizer=None, verbose=True, X_val=None, y_val=None):
        samples = len(X) # num_samples
        for epoch in range(1, epochs + 1):
            # Shuffle dataset, creates shuffled X and y to avoid bias from odering
            indices = np.arange(samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Determine batch size
            batch_size = batch_size or samples

            # Mini-batch training
            for start in range(0, samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward + loss
                loss = self.forward(X_batch, y_batch)

                # Backward
                self.backward(y_batch)

                # Update parameters
                optimizer.pre_update_params()
                for layer in self.layers:
                    if isinstance(layer, Layer_Dense):
                        optimizer.update_params(layer)
                optimizer.post_update_params()

            # Verbose logging
            if verbose:
                # Training predictions
                output_train = self.forward(X)
                predictions = np.argmax(output_train, axis=1)
                # if y is one hot encoded, convert to class labels
                y_true = np.argmax(y, axis=1) if len(y.shape) == 2 else y
                acc = np.mean(predictions == y_true)

                msg = f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}"

                # Validation
                if X_val is not None and y_val is not None:
                    output_val = self.forward(X_val)
                    val_pred = np.argmax(output_val, axis=1)
                    y_val_true = np.argmax(y_val, axis=1) if len(y_val.shape) == 2 else y_val
                    val_acc = np.mean(val_pred == y_val_true)
                    val_loss = self.layers[-1].loss.calculate(output_val, y_val)
                    msg += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"

                print(msg)
