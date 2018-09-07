import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

feta_acc = np.load('cifar10/results/accuracy/no_redundant_feta_acc.npy')
feta_sp = np.load('cifar10/results/accuracy/no_redundant_feta_spars.npy')
thr_acc = np.load('cifar10/results/accuracy/no_redundant_thresh_acc.npy')
thr_sp = np.load('cifar10/results/accuracy/no_redundant_thresh_spars.npy')

size_font = 12

fig2, ax2 = plt.subplots()
plt.ylabel('Accuracy %',fontsize=size_font)
plt.xlabel('Sparsity %',fontsize=size_font)

ax2.set_ylim(ymin=0, ymax=60)
ax2.set_xlim(xmin=45, xmax=100)



ax2.plot(feta_sp,feta_acc,linestyle = '-',color = 'xkcd:red')
ax2.plot(thr_sp,thr_acc,linestyle = '-',color = 'xkcd:bright blue')


patch0 = mpatches.Patch(color = 'xkcd:red', label = "FeTa")
patch1 = mpatches.Patch(color = 'xkcd:bright blue', label = "Thresholding")

plt.legend(loc = 2,handles=[patch0,patch1],fontsize=size_font)
plt.grid(linestyle=':')

end = 1