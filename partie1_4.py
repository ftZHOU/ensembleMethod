import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn import svm
import tools
import numpy as np

my_sampler = lambda n: make_circles(n, noise=0.1, factor=0.75)

clf = svm.NuSVC(gamma=1.0)
X, y = my_sampler(100)
clf.fit(X, y)
tools.plot_decision_boundary(clf, (-2, 2), (-2, 2))

# Plot the training set
colors = np.array(['#377eb8','#ff7f00'])
plt.scatter(X[:,0], X[:,1], c=colors[y])
plt.show()