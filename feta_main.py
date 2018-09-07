import numpy as np

# Here we define a number of functions used by FeTa.

# 1) FastNetTrim: contains initializations of the main variables and the outer loops of the algorithm.
# 2) minibatch: generates a minibatch from the data
# 3) gradient_h: computes the gradient of the concave part of the objective.
# 3) SVRG_minibatch_acc: performs SVRG algorithm from A PROXIMAL STOCHASTIC GRADIENT METHOD WITH PROGRESSIVE VARIANCE REDUCTION with acceleration.
# 5) objective: evaluates the data fidelity term of the objective using the current value of U.
# 4) prox_l1: performs the proximal step for sparsifying the matrix U.
# 6) cal_metrics: evaluates the sparsity of the matrix U.
# 7 grad_apr: this is a script for approximating the gradient of the rectifier. we started originally from an approximation as well but note that q(x)=exp(x)/(1+exp(x)) is numerically unstable for large values in X (overflow). We can instead replace it with 1 for x>=5 0 for x<-5 and exp(x)/(1+exp(x)) for other values
# 8) add_bias: adds values of 1 to the latent representations. This is needed to compute a layer bias.

#Inputs:
# Xtr: The latent input representations for the training set.
# Ytr: The latent output representations for the training set.
# Xval: The latent input representations for the validation set.
# Yval: The latent output representations for the validation set.
# params: A parameter vector for controlling the optimisation.

#Outputs:
# U: The pruned layer, the first row U[0,:] is the layer bias.
# obj: The values of the objective per iteration.
# Usparsity: The sparsity level per iteration.

def FastNetTrim(Xtr,Ytr,Xval,Yval,params):

    #initialize obj
    obj = np.empty([int(params[0])])
    Usparsity = np.empty([int(params[0])])

    Xtr = add_bias(Xtr)
    Xval = add_bias(Xval)

    dimX = Xtr.shape[0]
    dimY = Ytr.shape[0]


    #initialize U
    U = np.random.normal(0,0.001,(dimX,dimY)) #Concerning matrix dimensions, U.T*X is a valid multiplication.

    for i in range(0,int(params[0]) ):



        # Compute gradient of concave part Maybe redundant??
        grad_H = gradient_h(U,Xtr,Ytr,params)


        # Perform gradient step on linearized objective

        U = SVRG_minibatch_acc(U,grad_H,Xtr,Ytr,params)

        # Compute Objective at the given location
        obj[i] = objective(U,Xval,Yval,params)
        Usparsity[i] = cal_metrics(U,params)

        print("Iteration:", i,":: obj:",obj[i],":: sparsity:",Usparsity[i])



    return(U,obj,Usparsity)

def minibatch(Xts,Yts,params):

    indices = np.random.randint(0,Xts.shape[1],int(params[4]))

    X = Xts[:,indices]
    Y = Yts[:, indices]

    return(X,Y)

def gradient_h(U,X,Y,params):
    theta = params[1]
    grad_H = 2 * (np.maximum(Y, 0)) * grad_apr(theta*(U.T @ X ) ) @ X.T * (1 / params[7])
    return(grad_H)


def SVRG_minibatch_acc(U,grad_H,Xts,Yts,params):

    s_size = 3
    hta = params[5]
    theta = params[1]
    X_til_s = U
    b = params[4]
    n = params[7]
    m = int(n/b)

    beta = 0.95

    for s in range(0,s_size):

        print("Outer Iteration: ",s)

        X_til = X_til_s
        grad_til = 2 * ((1/theta)*np.log( 1+np.exp( theta*(X_til.T @ Xts) ) )) * grad_apr(theta*(X_til.T @ Xts) ) @  Xts.T * (1/params[7])
        X_i = X_til
        Y_i = X_til

        for i in range(0,m):

            X, Y = minibatch(Xts, Yts, params)

            com1 = 2 * ((1 / theta) * np.log(1 + np.exp(theta * (Y_i.T @ X)))) * grad_apr(theta * (Y_i.T @ X))*(1/params[4])

            com2 = 2 * ((1/theta)*np.log( 1+np.exp( theta*(X_til.T @ X) ) )) * grad_apr(theta*(X_til.T @ X) )*(1/params[4])

            grad_i = (com1-com2)@ X.T+grad_til

            tmp = Y_i-hta*(grad_i.T-grad_H.T)

            X_i_1 = prox_l1(tmp,params[8])

            Y_i = X_i_1 + beta*(X_i_1-X_i)

            X_i = X_i_1

        X_til_s = X_i

    return X_til_s


def objective(U,X,Y,params):
    obj = np.linalg.norm( (1/params[1])*np.log(1+params[1]*np.exp(U.T @ X) ) - Y)

    obj = (obj*obj)/X.shape[1]

    return obj

def prox_l1(U,spL):
    U = np.maximum(0, np.abs(U) - spL) * np.sign(U)
    return U

def cal_metrics(U, params):
    loc  = np.where(np.abs(U) == 0)
    Usparsity = (loc[0].shape[0])/(U.shape[0]*U.shape[1])*100

    return(Usparsity)

def grad_apr(x):

    pos_1 = np.where(x>=10)
    pos_0 = np.where(x<=-10)
    pos_other = np.where(x<10) and np.where(x>-10)
    #Set ones
    x[pos_1]=1
    #Set zeros
    x[pos_0]=0
    #Set other values
    x[pos_other]= np.exp(x[pos_other])*np.reciprocal(1+ np.exp(x[pos_other]) )

    return x

def add_bias(X):
    X = np.vstack( ( np.ones((1,X.shape[1])) ,X ) )
    return(X)
