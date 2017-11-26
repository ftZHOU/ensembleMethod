from sklearn.datasets import make_moons
import numpy as np 
import matplotlib.pyplot as plt
import tools
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier



my_sampler = lambda n: make_moons(n, shuffle=True, noise=0.25, random_state=None)

my_classifiers = [tree.DecisionTreeClassifier(), BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(), n_estimators=50),AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=1),n_estimators=50)]			



tools.fit_evaluate_plot_classifiers(my_classifiers, my_sampler,
                                    list(range(50, 1000, 50)), 100,
                                    nruns=30,
                                    title="BaggingClassifier",
                                    labels=['Decision tree', 'BaggingClassifier','BoostingClassifier'],
                                    ymax=0.55)
plt.show()
