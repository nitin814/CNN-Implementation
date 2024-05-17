import numpy as np

class layers:
    def __init__ (self, inputs, neurons , activation):
        self.inputs = inputs
        self.neurons = neurons
        self.activation = activation
        self.weights = np.random.uniform(-0.5 , 0.5, size=(neurons, inputs)).astype(np.float32)
        self.bias = np.random.uniform(-0.5, 0.5, size=(neurons, 1)).astype(np.float32)
        self.output = np.zeros(neurons)
        self.error = np.zeros(neurons)
        self.output = self.output.reshape((neurons , 1)).astype(np.float32)
        self.error = self.error.reshape((neurons , 1)).astype(np.float32)  

    def activations (self , x):
        return self.sigmoid(x);
        
    def sigmoid (self , x):
        return 1/(1+np.exp(-x))
    
    def relu (self , x):
        return max(0 , x)
    
    def outputs (self , inputs):  # we will get outputs of size of (neurons , 1) 
        self.output = self.activations(np.dot(self.weights , inputs) + self.bias)
        return self.output
    
    def errors (self , targets , isEnd): ## we will get error of size of (neurons , 1)
        if isEnd:
            self.error = (self.output)*(1-self.output)*(targets - self.output)
        else: 
            self.error = (self.output)*(1-self.output)*(targets)
        return np.dot(self.weights.T , self.error)
    
    def update_weights (self , input , learning_rate = 0.02):
        self.weights += learning_rate * np.dot(self.error , input.T)
        self.bias += learning_rate * self.error
        return self.output
    

class ann:
    def __init__ (self , learning_rate = 0.1 , epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layers = [];

    def add_layer(self, layer):
        self.layers.append(layer)

    def feed_forward (self , inputs):
        for layers in self.layers:
            inputs = layers.outputs(inputs)
        return inputs

    def back_propagate (self , inputs , target): # inputs are of size (1 , inputs)
        flag = 0
        for layers in reversed(self.layers):
            if flag == 0:
                target = layers.errors(target , 1)
                flag = 1
            else:
                target = layers.errors(target , 0)

        for layers in self.layers:
            inputs = layers.update_weights(inputs)
        
        return np.dot(self.layers[0].weights.T , self.layers[0].error)