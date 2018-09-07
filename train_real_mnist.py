# Simple model for Mnist
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

#This code trains a fully connected DNN to classify the Mnist dataset. For the output dimensions of the fully connected layers
# we have a default setting of:
# Dense1.out = 1000
# Dense2.out = 500
# Dense3.out = #classes
# which gets an accuracy of 98%.

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Create the model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dropout(0.2))
model.add(Dense(60, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(60, activation='relu', kernel_constraint=maxnorm(3)))
#model.add(Dropout(0.2))
#model.add(Dense(250, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
epochs = 30
lrate = 0.001
decayy = lrate/epochs
adam = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decayy, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=300)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))#98%

model.save('mnist/mnist_model.h5')

end  = 1