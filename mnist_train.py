# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from keras.utils import np_utils
get_ipython().magic('matplotlib inline')


# Importing data from csv files:
# 1. train.csv : The training data set, has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.
# 2. test.csv: The test data set, (test.csv), is the same as the training set, except that it does not contain the "label" column.
#
# Reading train data using pandas.
#
#

# In[2]:


data = pd.read_csv("../input/train.csv")
data = data.values
#Taking labels(first column) out of data.
label = data[:,0]

# Drop 'label' column
data = data[:,1:]

print("Data loaded, ready to go!")


# In[3]:


#plot distribution of label values
g = sns.countplot(label)


# Diving data and label into two parts: train and validation.
# We have a separate test.csv file for testing our model predictions

# In[4]:


#splitting data into train and valid
train_data=data[:35000,:]
valid_data=data[35000:,:]

#reshaping to make it in proper input shape for a neural network
train_data = train_data.reshape(train_data.shape[0], 1, 28, 28).astype('float32')
valid_data = valid_data.reshape(valid_data.shape[0], 1, 28, 28).astype('float32')

#normalise data
train_data = train_data / 255
valid_data= valid_data/255

#spliting label into train and valid
train_label = label[:35000]
valid_label = label[35000:]

#one-hot-encoding
#Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
train_label = np_utils.to_categorical(train_label)
valid_label = np_utils.to_categorical(valid_label)

#print shape
print("train_data shape: ",train_data.shape)
print("train_label shape: ",train_label.shape)
print("valid_data shape: ",valid_data.shape)
print("valid_label shape: ",valid_label.shape)


# Importing modules needed to build model
# Keras does provide a lot of capability for creating convolutional neural networks.

# In[5]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# Define a function to create a model

# In[6]:


def create_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Passing training and validation data along with labels to model

# In[7]:


# build the model
model = create_model()
# Fit the model
model.fit(train_data, train_label, validation_data=(valid_data, valid_label), epochs=10, batch_size=200, verbose=2)


# In[8]:


# Final evaluation of the model
scores = model.evaluate(valid_data, valid_label, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))


# Exciting! we have trained our model
# Saving model weights for later use

# In[9]:


model.save("model.h5")
print("model weights saved in model.h5 file")


# Saving model information in .json file

# In[10]:


from keras.models import model_from_json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print("model saved as model.json file")
