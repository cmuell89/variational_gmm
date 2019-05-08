import os
import json
import random
import numpy as np
from vbgmm_precision import VariationalGMM

def import_demonstrations(path_to_demos, count=None, random=False):

    demo_folders = os.listdir(path)
    if random and count:
        demo_folders = demo_folders[random.randrange(count)]
    elif count > 0 or count is not None:
        demo_folders = demo_folders[:count]

    demonstrations = []
    for i in demo_folders:
        demo = []
        for filename in os.listdir(os.path.join(path, i)):
            if filename == 'joint_angles.txt':
                with open(os.path.join(path, i, filename)) as file:
                    for line in file.readlines():
                        demo.append(json.loads(line))
        demonstrations.append(demo)

    return demonstrations

def vectorize_demonstrations(demonstrations):
    data_keys = ['left_e0', 'left_e1', 'left_s0', 'left_s1','left_w0', 'left_w1', 'left_w2', 'right_e0',
     'right_e1', 'right_s0','right_s1', 'right_w0', 'right_w1', 'right_w2', 'torso_t0']

    vectorized_demonstrations = np.array([])
    for demo in demonstrations:
        vectorized_demo = np.array([])
        for entry in demo:
            vector = np.array([])
            # append adds to end, so reverse list to retain original order
            for key in reversed(data_keys):
                np.append(vector, entry[key])
            np.append(vectorized_demo, vector)
        np.append(vectorized_demonstrations, vectorized_demo)
    return vectorized_demonstrations


if __name__ == "__main__":
    path = "/home/carlmueller/coursework/6519/data"
    demonstrations = vectorize_demonstrations(import_demonstrations(path, 5))
    print(demonstrations)

    import numpy as np
    import numpy.linalg as linalg
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # your ellispsoid and center in matrix form
    A = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 2]])
    center = [0, 0, 0]

    # find the rotation matrix and radii of the axes
    U, s, rotation = linalg.svd(A)
    radii = 1.0 / np.sqrt(s)

    # now carry on with EOL's answer
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color='b', alpha=0.2)
    plt.show()
    plt.close(fig)
    del fig