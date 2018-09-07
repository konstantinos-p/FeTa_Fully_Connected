# FeTa_Fully_Connected
Experiments comparing FeTa and Hard Thresholding for fully connected architectures. We test fully connected architectures for three datasets Mnist, FashionMnist and Cifar10. 

<B>train\_\*</B>: execute training of the DNNs. 
<B>plot\_\*</B>: plot accuracy (%) vs sparsity (%) for the corresponding architecture and the two methods (FeTa and Hard Thresholding).

<B>feta\_\*\_accuracy</B>: compute the accuracy vs sparsity of FeTa for different architectures.
<B>feta\_\*\_l2_norm</B>: compute the l_2 norm between the unpruned and pruned representations vs sparsity (%)for FeTa and different architectures.   

<B>thresholding\_\*\_accuracy</B>: compute the accuracy vs sparsity of Hard Thresholding for different architectures.
<B>thresholding\_\*\_l2_norm</B>: compute the l_2 norm between the unpruned and pruned representations vs sparsity (%) for Hard Thresholding and different architectures.   

<B>AB_matrices\_\*</B>: generate the intermediate representations needed to train FeTa, for different architectures.

<B>fera_main.py</B>: corresponds to the implementation of the FeTa algorithm.

<B>utils.py</B>: includes the implementation of the Hard Thresholding algorithm.
