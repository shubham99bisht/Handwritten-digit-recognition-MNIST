
# Handwritten Digit Recognition using Convolutional Neural Networks in Python with Keras

The “hello world” of object recognition for machine learning and deep learning is the MNIST dataset for handwritten digit recognition.

In this post you will discover how to develop a deep learning model to achieve near state of the art performance on the MNIST handwritten digit recognition task in Python using the Keras deep learning library.

After completing this tutorial, you will know:

How to load the MNIST dataset in Keras.
How to develop and evaluate a baseline neural network model for the MNIST problem.
How to implement and evaluate a simple Convolutional Neural Network for MNIST.
How to implement a close to state-of-the-art deep learning model for MNIST.


### Let's get started:

#### Description of the MNIST Handwritten Digit Recognition Problem

The MNIST problem is a dataset developed by Yann LeCun, Corinna Cortes and Christopher Burges for evaluating machine learning models on the handwritten digit classification problem.

The dataset was constructed from a number of scanned document dataset available from the National Institute of Standards and Technology (NIST). This is where the name for the dataset comes from, as the Modified NIST or MNIST dataset.

Images of digits were taken from a variety of scanned documents, normalized in size and centered. This makes it an excellent dataset for evaluating models, allowing the developer to focus on the machine learning with very little data cleaning or preparation required.

Each image is a 28 by 28 pixel square (784 pixels total). A standard spit of the dataset is used to evaluate and compare models, where 60,000 images are used to train a model and a separate set of 10,000 images are used to test it.

It is a digit recognition task. As such there are 10 digits (0 to 9) or 10 classes to predict. Results are reported using prediction error, which is nothing more than the inverted classification accuracy.

Excellent results achieve a prediction error of less than 1%. State-of-the-art prediction error of approximately 0.2% can be achieved with large Convolutional Neural Networks.



```python
uploaded = "/train.csv"
import pandas as pd
import io
data = pd.read_csv("train.csv")
#io.StringIO(uploaded.decode('utf-8'))
data= data.values
```


```python
from keras.utils import np_utils
print(data[0][:15])
print(data.shape)

label=data[:,0]

train_data=data[:35000,1:]
valid_data=data[35000:40000,1:]
test_data=data[40000:,1:]


train_data = train_data.reshape(35000, 1, 28, 28).astype('float32')
valid_data = valid_data.reshape(valid_data.shape[0], 1, 28, 28).astype('float32')
test_data = test_data.reshape(test_data.shape[0], 1, 28, 28).astype('float32')

"""
train_data = train_data.astype('float32')
valid_data = valid_data.astype('float32')
test_data = test_data.astype('float32')
"""

train_data = train_data / 255
valid_data= valid_data/255
test_data = test_data / 255

train_label=label[:35000]
valid_label=label[35000:40000]
test_label=label[40000:]

train_label = np_utils.to_categorical(train_label)
valid_label = np_utils.to_categorical(valid_label)
test_label = np_utils.to_categorical(test_label)


print(train_data.shape, test_data.shape, valid_data.shape)
print(train_label.shape, test_label.shape, valid_label.shape)
```

    Using TensorFlow backend.


    [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    (42000, 785)
    (35000, 1, 28, 28) (2000, 1, 28, 28) (5000, 1, 28, 28)
    (35000, 10) (2000, 10) (5000, 10)



```python
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
```


```python
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```


```python
# Fit the model
model.fit(train_data, train_label, validation_data=(valid_data, valid_label), epochs=1, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(valid_data, valid_label, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
```

    Train on 35000 samples, validate on 5000 samples
    Epoch 1/1
     - 97s - loss: 0.1184 - acc: 0.9657 - val_loss: 0.0850 - val_acc: 0.9738
    Baseline Error: 2.62%

