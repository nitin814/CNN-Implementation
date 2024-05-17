import numpy as np

class conv2d :
    def __init__ (self , image_depth , output_size , kernal_dim , padding = 1 , stride = 1):
        self.image_depth = image_depth
        self.output_size = output_size
        self.kernal_dim = kernal_dim
        self.padding = padding
        self.kernal = np.random.uniform(-0.5 , 0.5 , size = (output_size , image_depth , kernal_dim , kernal_dim)).astype(np.float32)
        self.bias = np.random.uniform(-0.5 , 0.5 , size = (output_size)).astype(np.float32)
        self.stride = stride

    def activation (self , x):
        return np.maximum(x , 0)   
    
    def forward (self , image):
        self.input = image
        output = np.zeros((self.output_size , image.shape[1] - self.kernal_dim + 1 , image.shape[2] - self.kernal_dim + 1))
        for i in range(self.output_size): # assuming that stide is 1 , for now 
            for j in range(image.shape[1] - self.kernal_dim + 1):
                for k in range(image.shape[2] - self.kernal_dim + 1):
                    image_slice = image[:,j : j + self.kernal_dim , k : k + self.kernal_dim]
                    output[i,j,k] = np.sum(image_slice * self.kernal[i , : , : , :]) + self.bias[i]
        return self.activation(output)
    
    def backward (self , error_layer , learning_rate = 0.02):
        error = np.zeros(self.input.shape)
        for i in range(self.output_size):
            for k in range(self.kernal_dim):
                for l in range(self.kernal_dim):
                    error[:,k : k + self.kernal_dim , l : l + self.kernal_dim] += error_layer[i , k , l] * self.kernal[i , : , : , :]
        
        for i in range(self.output_size):
            for k in range(self.kernal_dim):
                for l in range(self.kernal_dim):
                    image_slice = self.input[:,k : k + self.kernal_dim , l : l + self.kernal_dim]
                    self.kernal[i , : , : , :] -= learning_rate * error_layer[i , k , l] * image_slice
                    self.bias[i] -= learning_rate * error_layer[i , k , l]
        
        return error 
        

class max_pooling :
    def __init__ (self , pool_shape , stride = 1):
        self.pool_shape = pool_shape
        self.stride = stride

    def forward (self , image):
        self.input = image  
        output = np.zeros((image.shape[0] , image.shape[1] - self.pool_shape + 1 , image.shape[2] - self.pool_shape + 1))
        for i in range(image.shape[0]):
            for k in range(image.shape[1] - self.pool_shape + 1): # assuming that stride is 1 for now ...
                for l in range(image.shape[2] - self.pool_shape + 1): # assuming that stride is 1 for now ...
                    image_slice = image[i , k : k + self.pool_shape , l : l + self.pool_shape]
                    output[i , k , l] = np.max(image_slice)
        return output    

    def backward (self , error_layer , learning_rate = 0.02):
        image = self.input
        image_error = np.zeros(image.shape)

        for j in range(error_layer.shape[0]):
            for k in range(error_layer.shape[1]):
                for l in range(error_layer.shape[2]):
                    image_slice = image[j , k : k + self.pool_shape , l : l + self.pool_shape]
                    max_val = np.max(image_slice)
                    image_error[j , k : k + self.pool_shape , l : l + self.pool_shape] = (image_slice == max_val) * error_layer[j , k , l]
        
        return image_error
