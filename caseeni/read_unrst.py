import os
import numpy as np
import pdb

from ecl import EclipseFile

os.system("clear")


def read_and_save_pressures(data_folder, file_name, timestep):
    """ """
    with EclipseFile(file_name, "UNRST") as store:
        total = store.cnt["SEQNUM  "]  ### num timesteps ? and why does it have spaces??

        for i in range(0, total):
            time = i * timestep
            np.save(
                data_folder + "/fluid_pressure_" + str(time),
                store.get("PRESSURE", seq=i),
            )


def pressure_echelon_to_numpy():
    """ """
    
    data_folder_root = "./data"
    data_folder_fluid = data_folder_root + "/fluid"
    timestep = np.loadtxt(data_folder_root + "/TIMESTEP")

    training_dataset_id = np.loadtxt(data_folder_root + "/training_dataset_id")
    validation_dataset_id = np.loadtxt(data_folder_root + "/validation_dataset_id")
    test_dataset_id = np.loadtxt(data_folder_root + "/test_dataset_id")

    all_idx_mu = np.concatenate(
        (training_dataset_id, validation_dataset_id, test_dataset_id)
    ).astype(np.int32)
    

    for idx_mu in all_idx_mu:
        print("reading UNRST of idx_mu = ", idx_mu)
        data_folder = data_folder_fluid + "/" + str(int(idx_mu))
        file_name = data_folder + "/case2skew"
        read_and_save_pressures(data_folder, file_name, timestep)
        
    print("pressure converted")


if __name__ == "__main__":
    # # basically for testing...
    # data_folder_root = "./data"
    # data_folder_fluid = data_folder_root + "/fluid"
    # timestep = np.loadtxt(data_folder_root + "/TIMESTEP")

    # idx_mu = 99999
    # data_folder = data_folder_fluid + "/" + str(idx_mu)
    # file_name = data_folder + "/case2skew"
    # read_and_save_pressures(data_folder, file_name, timestep)

    # print("pressure converted")


    pressure_echelon_to_numpy()
    
    