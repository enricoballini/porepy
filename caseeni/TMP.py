import numpy as np


from matplotlib import pyplot as plt

n = 200
_mu_params = np.zeros((n, 7))
for i in np.arange(n):
    _mu_params[i] = np.load("./data/mu_param_" + str(i) + ".npy")

plt.plot(_mu_params[:, 0], "o")
# plt.plot(_mu_params[:, 1], "o")

plt.show()
