# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import fashion_mnist
from keras import backend as K
K.set_image_dim_ordering('th')
import utils as ut
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
path_to_model = 'fashion_mnist/fashion_mnist_model.h5'

iteration_steps = 10

flag1 = 1
flag2 = 1

perCsp = np.linspace(0.5,0.95,iteration_steps)

acc = np.zeros((iteration_steps,1))

model1 = load_model(path_to_model)

dense_1 = model1.get_layer('dense_1')
weights1  = dense_1.get_weights()[0]
bias1 = dense_1.get_weights()[1]

dense_2 = model1.get_layer('dense_2')
weights2  = dense_2.get_weights()[0]
bias2 = dense_2.get_weights()[1]

#dense_3 = model1.get_layer('dense_3')
#weights3  = dense_3.get_weights()[0]
#bias3 = dense_3.get_weights()[0]


for i in range(0,iteration_steps):

    new_dense_1 = ut.compute_thresholding_sparsification(weights1, perCsp[i])
    new_dense_2 = ut.compute_thresholding_sparsification(weights2, perCsp[i])
    acc[i] =  ut.evaluate_cifar_keras(path_to_model,[new_dense_1,bias1],[new_dense_2,bias2],flag1,flag2,X_test,y_test)[1]
    print('Calculating Step: ', i  )

plt.plot(perCsp*100, acc*100)

np.save('fashion_mnist/results/accuracy/no_redundant_thresh_spars.npy',perCsp*100)
np.save('fashion_mnist/results/accuracy/no_redundant_thresh_acc.npy',acc*100)

end = 1