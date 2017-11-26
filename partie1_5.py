import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn import svm
import tools

my_sampler = lambda n: make_circles(n, noise=0.1, factor=0.75)

my_classifiers = [svm.LinearSVC(), svm.NuSVC(gamma=1.0), svm.NuSVC(gamma=1000)]

tools.fit_evaluate_plot_classifiers(my_classifiers, my_sampler,
                                    list(range(50, 500, 50)), 1000,
                                    nruns=50,
                                    title="SVC",
                                    labels=['SVC Linear', 'SVC RBF ' + r'$\gamma=1.0$',  'SVC RBF '+ r'$\gamma=10^3$'],
                                    ymax=0.55)
plt.show()