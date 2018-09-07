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

l2_avg = np.zeros((iteration_steps,1))
l2_var = np.zeros((iteration_steps,1))

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

latent_X_original = np.load('fashion_mnist/testing/dense_2_output_ts.npy')

for i in range(0,iteration_steps):

    new_dense_1 = ut.compute_thresholding_sparsification(weights1, perCsp[i])
    new_dense_2 = ut.compute_thresholding_sparsification(weights2, perCsp[i])

    l2_avg[i],l2_var[i] =  ut.evaluate_l2_norm_keras(path_to_model,[new_dense_1,bias1],[new_dense_2,bias2],flag1,flag2,X_test,latent_X_original)

    print('Calculating Step: ', i  )

#plt.plot(perCsp*100, acc*100)

np.save('fashion_mnist/results/l2_norm/no_redundant_thresh_spars.npy',perCsp*100)
np.save('fashion_mnist/results/l2_norm/no_redundant_thresh_l2_avg.npy',l2_avg)
np.save('fashion_mnist/results/l2_norm/no_redundant_thresh_l2_var.npy',l2_var)

end = 1