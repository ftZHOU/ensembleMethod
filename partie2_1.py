from sklearn.datasets import make_moons
import numpy as np 
import matplotlib.pyplot as plt
import tools
from sklearn import tree
from sklearn.ensemble import BaggingClassifier


my_sampler = lambda n: make_moons(n, shuffle=True, noise=0.25, random_state=None)
X, y = my_sampler(100)
#max_depth


clf = tree.DecisionTreeClassifier()		
#clf.fit(X,y)														# clf = BaggingClassifier.decision_function()
model = BaggingClassifier(base_estimator=clf)
model.fit(X,y)
tools.plot_decision_boundary(model, (-2, 3), (-2, 3))

# Plot the training set
colors = np.array(['#377eb8','#ff7f00'])
plt.scatter(X[:,0], X[:,1], c=colors[y])
plt.show()