from sklearn.datasets import make_moons
import numpy as np 
import matplotlib.pyplot as plt
import tools
from sklearn import tree

my_sampler = lambda n: make_moons(n, shuffle=True, noise=0.25, random_state=None)

clf = tree.DecisionTreeClassifier(max_depth=1)	
X, y = my_sampler(10000)
clf.fit(X, y)
tools.plot_decision_boundary(clf, (-3, 3), (-3, 3))

# Plot the training set
colors = np.array(['#377eb8','#ff7f00'])
plt.scatter(X[:,0], X[:,1], c=colors[y])
plt.show()