from keras.models import load_model
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
import numpy as np
from keras.models import Model

def evaluate_cifar_keras(path_to_model,new_dense_1,new_dense_2,flag1,flag2,X_test,y_test):

    #load data and preprocess
    #(X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # normalize inputs from 0-255 to 0.0-1.0
    X_test = X_test.astype('float32')
    X_test = X_test / 255.0
    # one hot encode outputs
    y_test = np_utils.to_categorical(y_test)

    # Load model and Test
    model = load_model(path_to_model)


    #Set Layers

    if flag1 == 1:
        dense_1 = model.get_layer('dense_1')
        dense_1.set_weights(new_dense_1)
    if flag2 == 1:
        dense_2 = model.get_layer('dense_2')
        dense_2.set_weights(new_dense_2)


    scores = model.evaluate(X_test, y_test, verbose=0)
    return scores

def evaluate_l2_norm_keras(path_to_model,new_dense_1,new_dense_2,flag1,flag2,X_test,latent_X_original):

    #load data and preprocess
    #(X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # normalize inputs from 0-255 to 0.0-1.0
    X_test = X_test.astype('float32')
    X_test = X_test / 255.0
    # one hot encode outputs


    # Load model and Test
    model = load_model(path_to_model)


    #Set Layers

    if flag1 == 1:
        dense_1 = model.get_layer('dense_1')
        dense_1.set_weights(new_dense_1)
    if flag2 == 1:
        dense_2 = model.get_layer('dense_2')
        dense_2.set_weights(new_dense_2)

    dense_2_layer = Model(inputs=model.input, outputs=model.get_layer('dense_2').output)

    latent_X_new = dense_2_layer.predict(X_test)

    diff = latent_X_original-latent_X_new.T

    norm_change = np.linalg.norm(diff,axis=1)/np.linalg.norm(latent_X_original,axis=1)

    mean_ch = np.mean(norm_change)
    var_ch = np.var(norm_change)
    #max_ch = np.max(norm_change)
    #min_ch = np.min(norm_change)

    return mean_ch,var_ch

def compute_thresholding_sparsification(W,perCsp):
    #takes as input a matrix thresholds it based on magnitude to required sparsity


    b = np.reshape(np.abs(W), (-1))
    hist, bin_edges = np.histogram(b, bins=100, density=True)
    hist = hist / np.sum(hist)
    cumulative = np.cumsum(hist)
    pos = np.where(cumulative >= perCsp)
    threshold = bin_edges[pos[0][0]]


    W[np.where(np.abs(W) < threshold)] = 0

    # Calculate Sparsity in sanity check
    pos2 = np.where(W == 0)
    sparse = pos2[0].shape[0] / (W.shape[0] * W.shape[1])
    print("Sparsified W to: ",sparse*100," sparsity")

    return W