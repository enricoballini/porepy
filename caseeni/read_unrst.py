import os
import numpy as np
import pdb

from ecl import EclipseFile

os.system("clear")


def read_and_save_pressures(file_name, timestep):
    """ """
    with EclipseFile(file_name, "UNRST") as store:
        total = store.cnt["SEQNUM  "]  ### num timesteps ? and why does it have spaces??

        for i in range(0, total):
            time = i * timestep
            np.save(
                data_folder + "/fluid_pressure_" + str(time),
                store.get("PRESSURE", seq=i),
            )


data_folder_root = "./data"
data_folder_fluid = data_folder_root + "/fluid"
timestep = np.loadtxt(data_folder_root + "/TIMESTEP")


if __name__ == "__main__":
    # basically for testing...
    id_mu = 99999
    data_folder = data_folder_fluid + "/" + str(id_mu)
    file_name = data_folder + "/PP"

else:
    training_dataset_id = np.loadtxt(data_folder_root + "/training_dataset_id")
    validation_dataset_id = np.loadtxt(data_folder_root + "/validation_dataset_id")
    test_dataset_id = np.loadtxt(data_folder_root + "/test_dataset_id")

    all_id_mu = np.concatenate(
        (training_dataset_id, validation_dataset_id, test_dataset_id)
    )

    for id_mu in all_id_mu:
        data_folder = data_folder_fluid + "/" + str(id_mu)
        file_name = data_folder + "/IDK_TODO"
        read_and_save_pressures(file_name, timestep)


print("pressure converted")
