import numpy as np
import tqdm
from dense_class import layers , ann 

class cnn ():
    def __init__ (self , image_size):
        self.image_size = image_size
        self.layers = []
        ann_network = ann();
        self.ann_network = ann_network
        self.output = []
        self.flag = 0
        
    def add (self , layer):
        self.layers.append(layer)

    def add_ann (self , x , y): 
        self.output.append((x,y))

    def flatten (self , image):
        self.image_shape_unflattened = image.shape
        x = image.flatten()
        x = x.reshape(*x.shape , 1)
        return x
    
    def deflattenerror (self , error):
        return error.reshape(self.image_shape_unflattened)
    
    def forward (self , image):
        for layer in self.layers:
            image = layer.forward(image)
        x = self.flatten(image)

        if self.flag == 0:    
            for i in range(len(self.output)):
                if i == 0:
                    self.ann_network.add_layer(layers(x.shape[0] , self.output[i][1] , "sigmoid"))
                else:
                    self.ann_network.add_layer(layers(self.output[i][0] , self.output[i][1] , "sigmoid"))
            self.flag = 1
        
        ann_network = self.ann_network.feed_forward(x)
        return ann_network , x # for having the input nodes for ann network , for getting output call predict function


    def backward (self , image , output_label):
        error_ann = self.ann_network.back_propagate(image , output_label)
        error_cnn = self.deflattenerror(error_ann)

        for layer in reversed(self.layers):
            error_cnn = layer.backward(error_cnn)        

    def train (self , image , output_label , epochs = 3):
        vals = image
        for i in range(epochs):
            _,image = self.forward(vals)
            self.backward(image , output_label)
    
    def fit(self, inputs, target, epochs):
        for epoch in tqdm.tqdm(range(epochs), desc="Epochs"):
            epoch_error = 0
            for i in tqdm.tqdm(range(len(inputs)), desc="Training Progress", leave=False):
                output, ann_input = self.forward(inputs[i])
                epoch_error += np.sum((output - target[i])**2)
                self.backward(ann_input, target[i])
            if(epoch%1 == 0):
                print(f"Epoch {epoch + 1} error: {epoch_error / len(inputs)}")
    
    def evaluate(self, x_test, y_test):
        correct = 0
        for i in range(len(x_test)):
            prediction = self.predict(x_test[i])
            predicted_class = np.argmax(prediction)
            actual_class = np.argmax(y_test[i])
            if predicted_class == actual_class:
                correct += 1
        accuracy = (correct / len(x_test))*100
        print(f"Accuracy on test set: {accuracy:.4f}%")

    def predict(self, inputs):
        output, _ = self.forward(inputs)
        return output
