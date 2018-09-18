# FeTa_Fully_Connected
Experiments comparing FeTa and Hard Thresholding for fully connected architectures. We test fully connected architectures for three datasets Mnist, FashionMnist and Cifar10. The paper "FeTa: A DCA Pruning Algorithm with Generalization Error Guarantees" for the FeTa algorithm can be found  in https://arxiv.org/pdf/1803.04239.pdf .

<B>train\_\*</B>: These functions execute training of the DNNs. 

<B>plot\_\*</B>: These functions plot accuracy (%) vs sparsity (%) for the corresponding architecture and the two methods (FeTa and Hard Thresholding).

<B>feta\_\*\_accuracy</B>: These functions compute the accuracy vs sparsity of FeTa for different architectures.

<B>feta\_\*\_l2_norm</B>: These functions compute the l_2 norm between the unpruned and pruned representations vs sparsity (%)for FeTa and different architectures.   

<B>thresholding\_\*\_accuracy</B>: These functions compute the accuracy vs sparsity of Hard Thresholding for different architectures.

<B>thresholding\_\*\_l2_norm</B>: These functions compute the l_2 norm between the unpruned and pruned representations vs sparsity (%) for Hard Thresholding and different architectures.   

<B>AB_matrices\_\*</B>: These functions generate the intermediate representations needed to train FeTa, for different architectures.

<B>fera_main.py</B>: These function corresponds to the implementation of the FeTa algorithm.

<B>utils.py</B>: This function includes the implementation of the Hard Thresholding algorithm.
