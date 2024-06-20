import os
import sys
import pdb
import numpy as np

import sub_model_fom_case_eni

os.system("clear")

data_folder_root = "./data"
data_folder_mech = "./data/mech"

training_dataset_id = np.loadtxt(data_folder_root + "/training_dataset_id")
validation_dataset_id = np.loadtxt(data_folder_root + "/validation_dataset_id")
test_dataset_id = np.loadtxt(data_folder_root + "/test_dataset_id")
all_dataset_id = np.concatenate(
    (training_dataset_id, validation_dataset_id, test_dataset_id)
).astype(np.int32)

times_mech = np.loadtxt(data_folder_root + "/TIMES_MECH")

model = sub_model_fom_case_eni.SubModelCaseEni()
model.subscript = ""
model.save_folder = "./CANCELLARE"
model.set_geometry()
model.set_geometry_part_2()
model.set_equation_system_manager()
model.create_variables()
sd = model.mdg.subdomains(dim=3)[0]

# make folders:
for idx in all_dataset_id:
    os.system("mkdir " + data_folder_mech + "/" + str(idx))

# times files:
for idx in all_dataset_id:
    os.system(
        "cp ./data/TIMES_MECH "
        + data_folder_mech
        + "/"
        + str(idx)
        + "/PRUNED_TRAINING_TIMES"
    )

    os.system(
        "cp ./data/TIMES_MECH " + data_folder_mech + "/" + str(idx) + "/TIMES_MECH"
    )


# add possible missing data in mechanics data folder:
volumes_subdomains = sd.cell_volumes
volumes_interfaces = np.array([])
vars_domain = np.array([0])
dofs_primary_vars = np.arange(0, sd.num_cells)
n_dofs_tot = np.array([sd.num_cells])


for idx in all_dataset_id:
    np.save(
        data_folder_mech + "/" + str(idx) + "/volumes_subdomains", volumes_subdomains
    )
    np.save(
        data_folder_mech + "/" + str(idx) + "/volumes_interfaces", volumes_interfaces
    )
    np.save(data_folder_mech + "/" + str(idx) + "/vars_domain", vars_domain)
    np.save(data_folder_mech + "/" + str(idx) + "/dofs_primary_vars", dofs_primary_vars)
    np.save(data_folder_mech + "/" + str(idx) + "/n_dofs_tot", n_dofs_tot)


# generate fake displacement:
pdb.set_trace()
fake_displacement = 0.777 * np.ones(sd.num_cells)
for idx in all_dataset_id:
    for time in times_mech:
        np.save(
            data_folder_mech + "/" + str(idx) + "/displacement_" + "%.10f" % time,
            fake_displacement,
        )

print("\nDone!")
