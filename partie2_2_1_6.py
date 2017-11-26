from sklearn.datasets import make_moons
import numpy as np 
import matplotlib.pyplot as plt
import tools
from sklearn import tree
from sklearn.ensemble import BaggingClassifier


my_sampler = lambda n: make_moons(n, shuffle=True, noise=0.25, random_state=None)
#X, y = my_sampler(10000)

my_classifiers = [tree.DecisionTreeClassifier()]			


tools.fit_evaluate_plot_classifiers(my_classifiers, my_sampler,
                                    list(range(50, 1000, 50)), 1000,
                                    nruns=30,
                                    title="Tree",
                                    labels=['Tree'],
                                    ymax=0.55)
plt.show()

