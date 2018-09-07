# FeTa_Fully_Connected
Experiments comparing FeTa and Hard Thresholding for fully connected architectures. We test fully connected architectures for three datasets Mnist, FashionMnist and Cifar10. 

Functions train\_\* execute training of the DNNs. 
Functions plot\_\* plot accuracy (%) vs sparsity (%) for the corresponding architecture and the two methods (FeTa and Hard Thresholding).

Functions feta\_\*\_accuracy compute the accuracy vs sparsity of FeTa for different architectures.
Functions feta\_\*\_l2_norm compute the l_2 norm between the unpruned and pruned representations vs sparsity (%)for FeTa and different architectures.   

Functions thresholding\_\*\_accuracy compute the accuracy vs sparsity of Hard Thresholding for different architectures.
Functions thresholding\_\*\_l2_norm compute the l_2 norm between the unpruned and pruned representations vs sparsity (%) for Hard Thresholding and different architectures.   

Functions AB_matrices\_\* generate the intermediate representations needed to train FeTa, for different architectures.
