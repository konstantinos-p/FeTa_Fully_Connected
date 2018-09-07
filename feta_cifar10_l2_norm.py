# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10
from keras import backend as K
K.set_image_dim_ordering('th')
import utils as ut
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import feta_main as feta
import matplotlib.patches as mpatches

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
path_to_model = 'cifar10/cifar10_model.h5'

#(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
#path_to_model = 'fashion_mnist/fashion_mnist_model.h5'

#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#path_to_model = 'mnist/mnist_model.h5'

iteration_steps = 10

flag1 = 1
flag2 = 1

l2_avg = np.zeros((iteration_steps,1))
l2_var = np.zeros((iteration_steps,1))


model1 = load_model(path_to_model)

dense_1 = model1.get_layer('dense_1')
weights1  = dense_1.get_weights()[0]
bias1 = dense_1.get_weights()[1]

dense_2 = model1.get_layer('dense_2')
weights2  = dense_2.get_weights()[0]
bias2 = dense_2.get_weights()[1]



#Parameters for feta

params1 = np.empty([9])
params1[1]=10 # theta parameter
params1[7]=X_train.shape[0] # total number of samples
params1[0]=4 # number of outer loops
params1[5]=0.001 # gradient step
params1[4]=200 # batch size

params2 = np.empty([9])
params2[1]=10 # theta parameter
params2[7]=X_train.shape[0] # total number of samples
params2[0]=4 # number of outer loops
params2[5]=0.001 # gradient step
params2[4]=200 # batch size



#Load A and B matrices for feta

Xtr1 = np.load('cifar10/training/flatten_output_tn.npy')
Ytr1 = np.load('cifar10/training/dense_1_output_tn.npy')
Xtr2 = np.load('cifar10/training/dense_1_output_tn.npy')
Ytr2 = np.load('cifar10/training/dense_2_output_tn.npy')

Xval1 = np.load('cifar10/training/flatten_output_val.npy')
Yval1 = np.load('cifar10/training/dense_1_output_val.npy')
Xval2 = np.load('cifar10/training/dense_1_output_val.npy')
Yval2 = np.load('cifar10/training/dense_2_output_val.npy')

latent_X_original = np.load('cifar10/testing/dense_2_output_ts.npy')

#Sparsity thresholding values

#perCsp1 = np.array([0.000015,0.0000002,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001])
#perCsp2 = np.array([0.0002,0.000000017,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001])

perCsp1 = np.linspace(0.0000002,0.000015,iteration_steps)
perCsp2 = np.linspace(0.000000017,0.0002,iteration_steps)

perCsp = np.zeros((iteration_steps,1))

Usparsity1 = np.zeros((iteration_steps,1))
Usparsity2 = np.zeros((iteration_steps,1))

for i in range(0,iteration_steps):

    params1[8] = perCsp1[i]
    params2[8] = perCsp2[i]

    U2, obj2, tmp_sparsity2 = feta.FastNetTrim(Xtr2,Ytr2,Xval2,Yval2,params2)
    U1, obj1, tmp_sparsity1 = feta.FastNetTrim(Xtr1,Ytr1,Xval1,Yval1,params1)

    Usparsity1[i] = tmp_sparsity1[-1]
    Usparsity2[i] = tmp_sparsity2[-1]

    perCsp[i] = (Usparsity1[i]*weights1.shape[0]*weights1.shape[1]/100 +Usparsity2[i]*weights2.shape[0]*weights2.shape[1]/100)/(weights1.shape[0]*weights1.shape[1] +weights2.shape[0]*weights2.shape[1])

    new_dense_1 = U1[1:,:]
    new_dense_2 = U2[1:,:]

    bias1 = U1[0,:]
    bias2 = U2[0,:]

    l2_avg[i],l2_var[i] =  ut.evaluate_l2_norm_keras(path_to_model,[new_dense_1,bias1],[new_dense_2,bias2],flag1,flag2,X_test,latent_X_original)

    print('Calculating Step: ', i  )


#---------Plot
#size_font = 12

#fig2, ax2 = plt.subplots()
#plt.ylabel('l2 Norm %',fontsize=size_font)
#plt.xlabel('Sparsity %',fontsize=size_font)

#ax2.set_ylim(ymin=0, ymax=60)
#ax2.set_xlim(xmin=45, xmax=100)

#ax2.plot(perCsp*100,l2_avg,linestyle = '--',color = 'xkcd:violet')

#patch0 = mpatches.Patch(color = 'xkcd:violet', label = "FeTa")

#plt.legend(loc = 2,handles=[patch0],fontsize=size_font)


#ax2.fill_between(perCsp*100,l2_max,l2_min,interpolate=True,facecolor='xkcd:brick red',alpha=0.3)

#plt.grid(linestyle=':')

#---------

np.save('cifar10/results/l2_norm/no_redundant_feta_spars.npy',perCsp*100)
np.save('cifar10/results/l2_norm/no_redundant_feta_l2_avg.npy',l2_avg)
np.save('cifar10/results/l2_norm/no_redundant_feta_l2_var.npy',l2_var)


end = 1