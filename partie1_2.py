import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles


my_sampler = lambda n: make_circles(n, noise=0.1, factor=0.75)

X, y = my_sampler(1000)

colors = np.array(['#377eb8','#ff7f00'])

plt.figure()
plt.scatter(X[:,0], X[:,1], c=colors[y])
plt.axes().set_aspect('equal', 'datalim')
plt.show()