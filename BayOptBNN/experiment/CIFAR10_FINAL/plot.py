import gzip
import pickle
import numpy as np
from pylab import grid
import matplotlib
# matplotlib.use('Agg') # do this before importing pyplot
from matplotlib import pyplot as plt
plt.ioff()


from pylab import savefig

filename = "gp_cifar10.pdf"


inputDir_full = "BO_GP-resultValue.pkl"
with open(inputDir_full, 'rb') as f:
    Xdata, best_Y,s_in_min= pickle.load(f)

full = True


nstart=0
n = Xdata.shape[0]
if full == True:
    aux = (Xdata[1:n,:]-Xdata[0:n-1,:])**2
    distances = np.sqrt(aux.sum(axis=1))
else:

    if n >= 60:
        nstart=n-60
        aux = (Xdata[nstart+1:n,:]-Xdata[nstart:n-1,:])**2
        distances = np.sqrt(aux.sum(axis=1))
    else:
        aux = (Xdata[1:n,:]-Xdata[0:n-1,:])**2
        distances = np.sqrt(aux.sum(axis=1))

## Distances between consecutive x's
plt.figure(figsize=(15,5))
plt.subplot(1, 3, 1)
plt.plot(range(n-nstart-1), distances)
plt.xlabel('Iteration')
plt.ylabel('d(x[n], x[n-1])')
plt.title('Distance between consecutive x\'s')
grid(True)

# Estimated m(x) at the proposed sampling points
plt.subplot(1, 3, 2)
plt.plot(range(n-nstart),best_Y[nstart:n])
plt.title('Value of the best selected sample')
plt.xlabel('Iteration')
plt.ylabel('Best y')
grid(True)

# Plot of the proposed v(x) at the proposed sampling points
plt.subplot(1, 3, 3)
plt.errorbar(range(n-nstart),[0]*(n-nstart) , yerr=s_in_min[nstart:n,0],ecolor='b', capthick=1)
plt.title('Predicted sd. in the next sample')
plt.xlabel('Iteration')
plt.ylim(0,max(s_in_min[nstart:n,0])+np.sqrt(max(s_in_min[nstart:n,0])))
plt.ylabel('CI (centered at zero)')
grid(True)
savefig(filename)
