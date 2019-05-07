import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

from vbgmm_precision import VariationalGMM

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def read_file(filename):
    # skip over file info to get to data
    file = open(filename, "r")
    for line in file:
        if line.find("eruptions waiting") != -1:
            break
    # get data
    data = []
    for line in file:
        nb_ligne, eruption, waiting = [float(x) for x in line.split()]
        data.append(eruption)
        data.append(waiting)
    file.close()

    data = np.asarray(data)
    data.shape = (data.size // 2, 2)

    return data


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(3, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())
    plt.title(title)


if __name__ == "__main__":
    path = "./old_faithful.txt"
    X = read_file(path)

    # Generate random sample, two components
    np.random.seed(0)
    C = np.array([[0., -0.1], [1.7, .4]])

    n_components = 4

    # Fit a Gaussian mixture with EM using five components
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='full').fit(X)
    print("Regular GMM")
    print(gmm.means_)
    print(gmm.covariances_)
    print(gmm.predict(X))
    plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
                 '{}-component Gaussian Mixture Mode'.format(n_components))

    vbgmm = VariationalGMM(n_components=n_components).fit(X)
    print("\n\nMy variational GMM")
    print(vbgmm.means_)
    print(vbgmm.covariances_)
    # print(vbgmm.mixture_density(X))
    print(vbgmm.predict(X))
    plot_results(X, vbgmm.predict(X), vbgmm.means_, vbgmm.covariances_, 1,
                 'Bayesian Gaussian Mixture Model')

    # Fit a Dirichlet process Gaussian mixture using five components
    dpgmm = mixture.BayesianGaussianMixture(n_components=n_components,
                                            covariance_type='full', max_iter=200,
                                            weight_concentration_prior_type='dirichlet_distribution').fit(X)
    print("\n\nDPGMM")
    print(dpgmm.means_)
    print(dpgmm.covariances_)
    print(dpgmm.predict(X))
    plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 2,
                 'Bayesian Gaussian Mixture Model')
    plt.show()
