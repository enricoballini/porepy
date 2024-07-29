import os
import sys
import pdb

import numpy as np
from torch import nn

from nnrom.utils.nn_utils import *


"""
"""


def count_trainable_params(nn_list: list[nn.Module], save_folder: str) -> None:
    """ """
    trainable_params_count = [None] * len(nn_list)
    for i, nn in enumerate(nn_list):
        trainable_params_count[i] = sum(
            p.numel() for p in nn.parameters() if p.requires_grad
        )

        np.savetxt(
            save_folder + "/trainable_weights_nn_" + str(i),
            np.array([trainable_params_count[i]]),
        )
    np.savetxt(
        save_folder + "/trainable_weights_tot", np.array([sum(trainable_params_count)])
    )


def encoder_decoder_blu(data_folder, num_params):
    """ """
    n_dofs_tot = np.load(data_folder + "/n_dofs_tot.npy")[0]
    size_hidden = 100  # 40
    latent_size = 5+1
    mu_param_size = num_params  # time included

    encoder = nn.Sequential(
        nn.Linear(n_dofs_tot, latent_size),
        nn.PReLU(),
    )
    for layer in encoder:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")

    decoder = nn.Sequential(
        nn.Linear(latent_size, size_hidden),
        nn.PReLU(),
        nn.Linear(size_hidden, size_hidden),
        nn.PReLU(),
        nn.Linear(size_hidden, size_hidden),
        nn.PReLU(),
        #
        nn.Linear(size_hidden, size_hidden),
        nn.PReLU(),
        nn.Linear(size_hidden, size_hidden),
        nn.PReLU(),
        nn.Linear(size_hidden, size_hidden),
        nn.PReLU(),
        #
        nn.Linear(size_hidden, n_dofs_tot),
        nn.PReLU(),
    )
    for layer in decoder:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")

    blu = nn.Sequential(
        nn.Linear(mu_param_size, size_hidden),
        nn.PReLU(),
        nn.Linear(size_hidden, size_hidden),
        nn.PReLU(),
        #
        nn.Linear(size_hidden, size_hidden),
        nn.PReLU(),
        nn.Linear(size_hidden, size_hidden),
        nn.PReLU(),
        #
        nn.Linear(size_hidden, latent_size),
        nn.PReLU(),
    )

    for layer in blu:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")

    return encoder, decoder, blu
