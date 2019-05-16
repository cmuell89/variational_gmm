import os
import json
from collections import OrderedDict
import glob
import codecs
import itertools
import errno
import random
from vbgmm_precision import VariationalGMM
from sklearn import mixture

import numpy as np
from scipy import linalg
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!


color_iter = itertools.cycle(['navy', 'black', 'cornflowerblue', 'r',
                              'darkorange', 'g', 'brown'])


def load_json_files(path, count=None):
    """
    Import JSON files as a Python dictionary from .json files in the directory signified by the path..

    Parameters
    ----------
    path : string
        Path of directory containing the ..json files.

    Returns
    -------
    entries : dict
        Dictionary representation of the JSON file.
    """

    entries = []
    files = glob.glob(path)
    if count is not None and count > 0:
        files = files[0:count]
    for name in files:
        try:
            with codecs.open(name, "r", 'utf-8') as f:
                file_data = json.load(f, object_pairs_hook=OrderedDict)
                entries.append(file_data)
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise  # Propagate other kinds of IOError.
    return entries

def vectorize_demonstrations(demonstrations):
    vectorized_demonstrations = []
    for demo in demonstrations:
        vectorized_demo = []
        for entry in demo:
            vector = entry['robot']['position']
            vectorized_demo.append(vector)
        vectorized_demo.reverse()
        vectorized_demonstrations.append(vectorized_demo)
    return vectorized_demonstrations


def plot_results(X, Y_, means, covariances):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):

        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        ax.scatter(X[Y_ == i, 0], X[Y_ == i, 1], X[Y_ ==i, 2], s=.8, color=color)

        # find the rotation matrix and radii of the axes
        U, s, rotation = linalg.svd(covar)
        radii = np.sqrt(s)
        print(radii)

        # now carry on with EOL's answer
        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + mean

        # plot
        ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color=color, alpha=0.2)


if __name__ == "__main__":
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'place_atop_demos')
    demos = vectorize_demonstrations(load_json_files(os.path.join(dir_path, '*.json'), count=5))
    print(len(demos))
    X = np.array([e for sl in demos for e in sl])


    vbgmm = VariationalGMM(n_components=10).fit(X)

    print("\n\nMy variational GMM")
    print(vbgmm.means_)
    print(vbgmm.covariances_)
    # print(vbgmm.mixture_density(X))
    print(vbgmm.predict(X))
    plot_results(X, vbgmm.predict(X), vbgmm.means_, vbgmm.covariances_)
    plt.title('Bayesian Gaussian Mixture Model on Shelf Placement Demonstration')
    plt.show()

    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pouring_demos')
    demos = vectorize_demonstrations(load_json_files(os.path.join(dir_path, '*.json')))
    X = np.array([e for sl in demos for e in sl])

    vbgmm = VariationalGMM(n_components=10).fit(X)
    print(vbgmm.means_)
    print(vbgmm.covariances_)
    # print(vbgmm.mixture_density(X))
    print(vbgmm.predict(X))
    plot_results(X, vbgmm.predict(X), vbgmm.means_, vbgmm.covariances_)
    plt.title('Bayesian Gaussian Mixture Model on a Pouring Demonstration')
    plt.show()