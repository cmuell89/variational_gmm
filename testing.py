import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from vbgmm import VariationalGMM

def read_file(filename):
    # skip over file info to get to data
    file = open(filename, "r")
    for line in file:
        if line.find("eruptions waiting") != -1:
            break
    # get data
    data =  []
    for line in file:
        nb_ligne, eruption, waiting = [float(x) for x in line.split()]
        data.append(eruption)
        data.append(waiting)
    file.close()

    data = np.asarray(data)
    print(data)
    data.shape = (data.size // 2, 2)

    return data

if __name__ == "__main__":

    path = "./old_faithful.txt"
    data = read_file(path)

    gmm = VariationalGMM(K=25, alpha_prior= 1*10^(-5), max_iter=1001)
