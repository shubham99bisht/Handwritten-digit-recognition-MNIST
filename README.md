# Handwritten-digit-recognition-MNIST
Handwritten Digit Recognition using Convolutional Neural Networks in Python with Keras

## Mnist dataset:

![alt text](https://github.com/shubham99bisht/Handwritten-digit-recognition-MNIST/blob/master/src/mnist-sample.png "MNIST")



This is a 5 layers Sequential Convolutional Neural Network for digits recognition trained on MNIST dataset. I choosed to build it with keras API (Tensorflow backend) which is very intuitive. Firstly, I will prepare the data (divide into train, validation and test sets) then i will focus on the CNN modeling and evaluation.

I achieved 98.51% of accuracy with this CNN trained on a GPU, which took me about a minute. If you dont have a GPU powered machine it might take a little longer, you can try reducing the epochs (steps) to reduce computation.

After we are satisfied with our model performance we head towards the next step which is taking input from live camera.

For step-by-step tutorial please refer to [wiki](https://github.com/shubham99bisht/Handwritten-digit-recognition-MNIST/wiki). It will take you through all the steps right from loading the data into code.
