from cnn_class import cnn
from dense_class import ann , layers
from convolution_block_class import conv2d , max_pooling
import numpy as np
import emnist 

def extract_data ():
    train_images, train_labels = emnist.extract_training_samples('digits')
    test_images, test_labels = emnist.extract_test_samples('digits')

    train_images_float32 = train_images.astype(np.float32)
    train_images_float32 /= 255.0
    test_images_float32 = test_images.astype(np.float32)
    test_images_float32 /= 255.0

    print("Train images shape (reshaped):", train_images_float32.shape)
    print("Test images shape (reshaped):", test_images_float32.shape)

    input_size = train_images_float32.shape[1]
    output_size = len(np.unique(train_labels))

    def one_hot_encode(labels, num_classes):
        num_samples = len(labels)
        one_hot_labels = np.zeros((num_samples, num_classes))
        for i in range(num_samples):
            one_hot_labels[i, labels[i]] = 1
        return one_hot_labels

    train_labels_onehot = one_hot_encode(train_labels, output_size)
    test_labels_onehot = one_hot_encode(test_labels, output_size)

    train_labels_onehot = train_labels_onehot.reshape(*train_labels_onehot.shape , 1)
    train_images_float32 = train_images_float32.reshape(240000 , 1 , 28 , 28)

    return train_images_float32, train_labels_onehot, test_images_float32, test_labels_onehot



def main():
    train_images_float32, train_labels_onehot, test_images_float32, test_labels_onehot = extract_data()

    cnnbro = cnn((1 , 28 , 28))
    cnnbro.add(conv2d(1 , 4 , 3))
    cnnbro.add(max_pooling(2))
    cnnbro.add(conv2d(4 , 8 , 3))
    cnnbro.add(max_pooling(2))
    cnnbro.add_ann("initial" , 20)
    cnnbro.add_ann(20 , 10)

    cnnbro.fit(train_images_float32[0:1000] , train_labels_onehot[0:1000] , 50)
    cnnbro.evaluate(test_images_float32[0:100] , test_labels_onehot[0:100])


if __name__ == "__main__":
    main()