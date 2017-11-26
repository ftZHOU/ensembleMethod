# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.datasets import make_circles
from sklearn import svm

def compute_risk(classifier, sampler, nb_samples):
    """
    Computes the risk with the binary loss on a dataset of size nb_samples

    Parameters
    ----------
    classifier : object
                 The classifier must provide a predict method

    sampler : function
              Applied on an int "n", returns X, y the samples and labels

    nb_samples : int
                 Number of samples on which to compute the risk

    Returns
    -------
    risk : float
           The computed risk of the classifier

    """
    X, y = sampler(nb_samples)
    ypred = classifier.predict(X)
    return sum(y != ypred)/float(nb_samples)

def fit_evaluate_classifier(classifier, sampler, nb_samples_train, nb_samples_test):
    """
    This fits a classifier on a dataset and evaluates its real risk
    
    with nb_samples_train samples

    Parameters
    ----------
    classifier : object
                 The classifier must provide the fit and predict methods
    
    sampler : function
              Applied on an int "n", returns X, y the samples and labels

    nb_samples_train : int
                       The number of samples on which to train the classifier
    
    nb_samples_test : int
                      The number of samples on which to estimate the risk of the fitted classifier
    
    Returns
    -------
    risk : float
           The estimated real risk of the fitted classifier

    """
    X, y = sampler(nb_samples_train)
    classifier.fit(X, y)
    risk = compute_risk(classifier, sampler, nb_samples_test)
    return risk

def fit_evaluate_plot_classifiers(lclassifiers, sampler,
                                  lsize_nb_samples_train, nb_samples_test,
                                  nruns=50, title=None, labels=None, ymax=None):
    """
    This fits and evaluate several classifiers on several independent runs and plot the estimated real risks.

    Parameters
    ----------
    lclassifiers : list of objects
                   The classifiers must provide the fit and predict methods

    sampler : function
              Applied to n (int), returns X, y  the samples and their labels

    lsize_nb_samples_train : list of int
                             The sizes of training set on which to fit the classifiers

    nb_samples_test : int
                      The number of independently drawn samples on which to evaluate the risk

    nruns : int
            The number of independent runs for evaluating the performance of a classifier on a given training set size

    title : optional string

    labels : optional list of strings
             If provided, the labels for tagging the classifiers

    ymax : optional float
           If not provided, the ylim is automatically adjusted
    """
    with_labels = labels is not None
    if(labels is not None and len(labels) != len(lclassifiers)):
        print("Error: if you provide labels, you must provide as many labels as classifiers")
        sys.exit(-1)


    if not with_labels:
        labels = [""] * len(lclassifiers)
        
    # Some predefined colors. If more are required, we generate them randomly
    # on the fly
    colors=['c','m','r', 'k']
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # We iterate over our collection of classifiers
    # Fit them several times on different dataset size
    # and evaluate their performance
    for iclf, clf in enumerate(lclassifiers):
        print("Classifier {} {}".format(iclf, labels[iclf]))
        risks = []
        # For every possible training set size
        for n in lsize_nb_samples_train:
            sys.stdout.write('\r n = {}'.format(n))
            sys.stdout.flush()
            # We run nruns independent runs
            r = [fit_evaluate_classifier(clf, sampler, n, nb_samples_test) for i in range(nruns)]
            risks.append(r)
        # The risks array below is of shape [len(lsize_nb_samples_train), nruns]
        risks = np.array(risks)
        mu_risk = risks.mean(axis=1)
        std_risk = risks.std(axis=1)
        if(iclf >= len(colors)):
            c = np.random.rand(3,1)
        else:
            c = colors[iclf]
        if(with_labels):
            ax.fill_between(lsize_nb_samples_train, mu_risk - std_risk,
                            mu_risk + std_risk, alpha = 0.5, color=c)
            ax.plot(lsize_nb_samples_train, mu_risk, 'o-', label=labels[iclf], color=c)
            #ax.errorbar(lsize_nb_samples_train, mu_risk, yerr=std_risk, fmt='o', label=labels[iclf])
        else:
            ax.fill_between(lsize_nb_samples_train, mu_risk - std_risk,
                            mu_risk + std_risk, alpha = 0.5, color=c)
            ax.plot(lsize_nb_samples_train, mu_risk, 'o-', label=labels[iclf], color=c)

        print("")

    # We slightly adjust the limits of the x, y axis
    # to better see the plots
    if ymax is not None:
        ax.set_ylim((0, ymax))
    else:
        ylim = ax.get_ylim()
        ax.set_ylim((0, ylim[1]))
    xlim = ax.get_xlim()
    ax.set_xlim((0, xlim[1]+50))

    
    # And finally set up the titles, labels, legend, ...
    plt.xlabel("Training set size")
    plt.ylabel("Real risk estimated on {} independent samples".format(nb_samples_test))
    if(title):
        plt.title(title)
    if with_labels:
        # Resize the graph in order to put the legend on the side
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))



def plot_decision_boundary(classifier, xbounds, ybounds, Nsamples=100, colors=np.array(['#377eb8','#ff7f00'])):
    """
    This computes and plot the decision boundary of a classifier
    It does not trigger plt.show(), you must call it on your own

    Parameters
    ----------
    classifier : object
                 It must provide a predict method

    xbounds : pair (tuple) of xmin, xmax
    ybounds : pair (tuple) of ymin, ymax

    Nsamples : number of subdivisions of the x and y axis
    
    colors : optional numpy array
             The colors to be used for plotting the samples and decision boundary

    """
    
    plt.figure()

    dx = (xbounds[1] - xbounds[0])/float(Nsamples)
    dy = (ybounds[1] - ybounds[0])/float(Nsamples)
    
    xx, yy = np.meshgrid(np.arange(xbounds[0], xbounds[1], dx),
                         np.arange(ybounds[0], ybounds[1], dy))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4,
                 levels=[0, 0.5, 1.0], colors=colors)
    plt.axes().set_aspect('equal', 'datalim')

    
if __name__ == '__main__':

    my_sampler = lambda n: make_circles(n, noise=0.1, factor=0.75)

    #####################
    ### Plot some samples
    print("Getting and plotting some samples")
    
    X, y = my_sampler(1000)

    colors = np.array(['#377eb8','#ff7f00'])
    
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=colors[y])
    plt.axes().set_aspect('equal', 'datalim')
    plt.savefig('circles.png', bbox_inches='tight')
    plt.show()

    ####################
    ### Fit a classifier
    X, y = my_sampler(1000)
    clf = svm.LinearSVC()
    clf.fit(X, y)
    
    ##############################################################
    ### Estimate its real risk by computing the empirical risk
    ### on a large dataset drawn independently of the training set
    X, y = my_sampler(100)
    clf = svm.LinearSVC()
    clf.fit(X, y)
    risk = compute_risk(clf, my_sampler, 10**4)
    print("Risk : {}".format(risk))

    clf = svm.NuSVC()
    risk = fit_evaluate_classifier(clf, my_sampler, 100, 10**4)
    print("Risk : {}".format(risk))

    ##################################
    ### Evaluating several classifiers
    my_classifiers = [svm.LinearSVC(), svm.NuSVC(gamma=1.0), svm.NuSVC(gamma=1000)]
    
    fit_evaluate_plot_classifiers(my_classifiers, my_sampler,
                                  list(range(50, 500, 50)), 1000,
                                  title="SVC",
                                  labels=['SVC Linear', 'SVC RBF ' + r'$\gamma=1.0$',  'SVC RBF '+ r'$\gamma=10^3$'],
                                  ymax=0.55)
    plt.savefig("classifiers.png", bbox_inches='tight')

    ##############################
    ##### Plot a decision boundary
    clf = svm.NuSVC(gamma=1.0)
    X, y = my_sampler(100)
    clf.fit(X, y)
    plot_decision_boundary(clf, (-2, 2), (-2, 2))
    # Plot some samples
    plt.scatter(X[:,0], X[:,1], c=colors[y])
    plt.savefig("decision_boundary.png", bbox_inches='tight')
    plt.show()
