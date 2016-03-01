import numpy as np
import matplotlib.pyplot as plt

xi = np.array([0., 0.5, 1.0])
yi = np.array([0., 0.5, 1.0])
zi = np.array([[0., 1.0, 2.0],
               [0., 1.0, 2.0],
               [-0.1, 1.0, 2.0]])

v = np.linspace(-.1, 2.0, 15, endpoint=True)
plt.contour(xi, yi, zi, 100, linewidths=0.5, colors='k')
plt.contourf(xi, yi, zi, 100, cmap=plt.cm.jet)
x = plt.colorbar(ticks=v)
print x
plt.show()