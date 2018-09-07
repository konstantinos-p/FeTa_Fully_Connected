# Simple CNN model for CIFAR-10
import numpy as np
import matplotlib.pyplot as plt
import feta_main as feta


#Setting the probem size
Training_size = 1000
Validation_size = 100
d_input = 200 # number of latent input dimensions
d_output = 100 # number of latent output dimensions

#Parameters for feta
params = np.empty([9])
params[1]=10 # theta parameter for controlling the smoothness of the approximation of the rectifier.
params[7]=Training_size # total number of training samples.
params[0]=200 # number of outer loops. For realistic experiments 4-15 iterations are sufficient.
params[5]=0.001 # gradient step.
params[4]=200 # batch size.
params[8] = 0.0004 #the lambda parameter controlling sparsity. Needs to be hand tuned.

#Creating the synthetic dataset
Xtr = np.random.normal(0,1,(d_input,Training_size))
Xval = np.random.normal(0,1,(d_input,Validation_size))

U_dense = np.random.normal(0,1,(d_output,d_input))
bias_dense = np.random.normal(0,1,(d_output,1))

Ytr  = np.maximum(U_dense@Xtr+bias_dense,0)
Yval = np.maximum(U_dense@Xval+bias_dense,0)


# Running FeTa
U, obj, tmp_sparsity = feta.FastNetTrim(Xtr,Ytr,Xval,Yval,params)

# Plotting the results

size_font = 12

fig1, ax1 = plt.subplots()
plt.ylabel('|| ||_2',fontsize=size_font)
plt.xlabel('Iterations #',fontsize=size_font)
ax1.plot(np.arange(0,params[0]),obj,linestyle = '-',color = 'xkcd:blue')
plt.grid(linestyle=':')
plt.title('FeTa || ||_2')

fig2, ax2 = plt.subplots()
plt.ylabel('Sparsity %',fontsize=size_font)
plt.xlabel('Iterations #',fontsize=size_font)
ax2.plot(np.arange(0,params[0]),tmp_sparsity,linestyle = '-',color = 'xkcd:red')
plt.grid(linestyle=':')
plt.title('FeTa Sparsity')

end = 1