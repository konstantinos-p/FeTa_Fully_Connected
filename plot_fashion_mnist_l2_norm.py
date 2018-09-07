import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

feta_l2_avg = np.load('fashion_mnist/results/l2_norm/no_redundant_feta_l2_avg.npy')
feta_l2_var = np.load('fashion_mnist/results/l2_norm/no_redundant_feta_l2_var.npy')
feta_sp = np.load('fashion_mnist/results/l2_norm/no_redundant_feta_spars.npy')

thresh_l2_avg = np.load('fashion_mnist/results/l2_norm/no_redundant_thresh_l2_avg.npy')
thresh_l2_var = np.load('fashion_mnist/results/l2_norm/no_redundant_thresh_l2_var.npy')
thresh_sp = np.load('fashion_mnist/results/l2_norm/no_redundant_thresh_spars.npy')

size_font = 12

fig2, ax2 = plt.subplots()
plt.ylabel('l2 norm change %',fontsize=size_font)
plt.xlabel('Sparsity %',fontsize=size_font)


#ax2.set_ylim(ymin=0, ymax=60)
#ax2.set_xlim(xmin=45, xmax=100)


ax2.plot(feta_sp,feta_l2_avg,linestyle = '-',color = 'xkcd:red')
ax2.plot(thresh_sp,thresh_l2_avg,linestyle = '-',color = 'xkcd:bright blue')


patch0 = mpatches.Patch(color = 'xkcd:red', label = "FeTa")
patch1 = mpatches.Patch(color = 'xkcd:bright blue', label = "Thresholding")



ax2.fill_between(feta_sp[:,0],feta_l2_avg[:,0]-feta_l2_var[:,0],feta_l2_avg[:,0]+feta_l2_var[:,0],interpolate=True,facecolor='xkcd:brick red',alpha=0.3)
ax2.fill_between(thresh_sp[:],thresh_l2_avg[:,0]-thresh_l2_var[:,0],thresh_l2_avg[:,0]+thresh_l2_var[:,0],interpolate=True,facecolor='xkcd:blue',alpha=0.3)


plt.legend(loc = 2,handles=[patch0,patch1],fontsize=size_font)
plt.grid(linestyle=':')

end = 1