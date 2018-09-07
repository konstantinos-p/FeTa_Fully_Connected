import numpy as np
from keras.models import load_model
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


#load data and preprocess
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Load model and Test
model = load_model('cifar10/cifar10_model.h5')

flatten_layer = Model(inputs=model.input,outputs=model.get_layer('flatten_1').output)
dense_1_layer = Model(inputs=model.input,outputs=model.get_layer('dense_1').output)
dense_2_layer = Model(inputs=model.input,outputs=model.get_layer('dense_2').output)
dense_3_layer = Model(inputs=model.input,outputs=model.get_layer('dense_3').output)

# Training and Validation
indices_for_validation = np.random.randint(0,50000,5000)


flatten_output = flatten_layer.predict(X_train)
flatten_output = flatten_output.T
np.save('cifar10/training/flatten_output_val.npy',flatten_output[:,indices_for_validation])
np.save('cifar10/training/flatten_output_tn.npy',flatten_output)

dense_1_output = dense_1_layer.predict(X_train)
dense_1_output = dense_1_output.T
np.save('cifar10/training/dense_1_output_val.npy',dense_1_output[:,indices_for_validation])
np.save('cifar10/training/dense_1_output_tn.npy',dense_1_output)

dense_2_output = dense_2_layer.predict(X_train)
dense_2_output = dense_2_output.T
np.save('cifar10/training/dense_2_output_val.npy',dense_2_output[:,indices_for_validation])
np.save('cifar10/training/dense_2_output_tn.npy',dense_2_output)

dense_3_output = dense_3_layer.predict(X_train)
dense_3_output = dense_3_output.T
np.save('cifar10/training/dense_3_output_val.npy',dense_3_output[:,indices_for_validation])
np.save('cifar10/training/dense_3_output_tn.npy',dense_3_output)

#Testing
flatten_output = flatten_layer.predict(X_test)
flatten_output = flatten_output.T
np.save('cifar10/testing/flatten_output_ts.npy',flatten_output)

dense_1_output = dense_1_layer.predict(X_test)
dense_1_output = dense_1_output.T
np.save('cifar10/testing/dense_1_output_ts.npy',dense_1_output)

dense_2_output = dense_2_layer.predict(X_test)
dense_2_output = dense_2_output.T
np.save('cifar10/testing/dense_2_output_ts.npy',dense_2_output)

dense_3_output = dense_3_layer.predict(X_test)
dense_3_output = dense_3_output.T
np.save('cifar10/testing/dense_3_output_ts.npy',dense_3_output)

end = 1
