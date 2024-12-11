

"""
il solution strategy del modello chiama after_simulation che salva le matrici e vettori necessari.
Poi come postprocess chiamo compute_fault_traction_fom
gli script completi li trovi sul mio github: https://github.com/enricoballini/porepy/tree/eni/caseenicut
vedi run_all.sh dentro quel repository
"""


def after_simulation(self) -> None:
        """
        questa è una funzione di default dei modelli di porepy che ho sovrascritto, viene chiamata nel solution strategy
        """

        subdomains_data = self.mdg.subdomains(return_data=True)
        subdomains = [subdomains_data[0][0]]
        sd = subdomains[0]
        data = subdomains_data[0][1]
        boundary_grids = self.mdg.boundaries(return_data=False)

        stress_tensor_grad = data[pp.DISCRETIZATION_MATRICES][self.stress_keyword][
            "stress"
        ] # Quelle ottenute con MPSA
        bound_stress = data[pp.DISCRETIZATION_MATRICES][self.stress_keyword][
            "bound_stress"
        ]  
        u_b_displ = self.bc_values_displacement(boundary_grids[0])
        u_b_stress = self.bc_values_stress(boundary_grids[0])

        u_b = (
            u_b_displ + u_b_stress
        )  # they should be exclusive, either displ or stress is != 0

        u_b_filled = np.zeros((self.nd, sd.num_faces))
        u_b_filled[:, sd.get_all_boundary_faces()] = u_b.reshape((3, -1), order="F")
        u_b_filled = u_b_filled.ravel("F")

        u = self.displacement(subdomains).evaluate(self.equation_system).val

        normal_area = sd.face_normals[:, self.fracture_faces_id][
            :, 0
        ]  # the fracture is planar, i take the first vecor as ref

        # need to save data for computing fault traction as post process
        np.save(self.save_folder + "/displacement" + self.subscript, u)
        np.save(
            self.save_folder + "/displacement_boundary" + self.subscript, u_b_filled
        )
        sp.sparse.save_npz(
            self.save_folder + "/stress_tensor_grad" + self.subscript,
            stress_tensor_grad,
        )
        sp.sparse.save_npz(
            self.save_folder + "/bound_stress" + self.subscript, bound_stress
        )
        np.save(self.save_folder + "/normal_area", normal_area)
        np.save(self.save_folder + "/face_area", sd.face_areas)
        np.save(self.save_folder + "/fracture_faces_id", self.fracture_faces_id)

        with open(self.save_folder + "/sd_fract.pkl", "wb") as fle:
            pickle.dump(self.sd_fract, fle)

        volumes_subdomains = np.repeat(
            self.sd_fract.cell_volumes, 3
        )  # [vol cell1, vol cell1, vol cell1, vol cell2, vol cell2, ...]
        volumes_interfaces = np.array([])
        vars_domain = np.array([0])
        dofs_primary_vars = np.array([np.arange(0, 3 * self.sd_fract.num_cells)])
        n_dofs_tot = np.array([3 * self.sd_fract.num_cells], dtype=np.int32)

        np.save(self.save_folder + "/volumes_subdomains", volumes_subdomains)
        np.save(self.save_folder + "/volumes_interfaces", volumes_interfaces)
        np.save(self.save_folder + "/vars_domain", vars_domain)
        np.save(self.save_folder + "/dofs_primary_vars", dofs_primary_vars)
        np.save(self.save_folder + "/n_dofs_tot", n_dofs_tot)
        print("end after_simulation")




def compute_fault_traction_fom(idx_mu):
    """ """
    save_folder = "./results/mech/" + str(idx_mu)

    friction_coeff = 0.45 # form spe paper

    with open("./data/mech/" + str(idx_mu) + "/sd_fract.pkl", "rb") as fle:
        sd_fract = pickle.load(fle)

    exporter = pp.Exporter(
        sd_fract,
        file_name="sd_fract",
        folder_name=save_folder,
    )

    u_ref = np.load(
        "./data/mech/" + str(idx_mu) + "/displacement_" + str(0.0) + ".npy"
    )
    u_b_filled_ref = np.load(
        "./data/mech/"
        + str(idx_mu)
        + "/displacement_boundary_"
        + str(0.0)
        + ".npy"
    )
    times_mech = np.loadtxt("./data/TIMES_MECH")

    def parallel_fcn(time):
        """ """
        time = time[0]
        print("time = ", time)
        
        stress_tensor_grad = sp.sparse.load_npz(
            "./data/mech/" + str(idx_mu) + "/stress_tensor_grad_" + str(time) + ".npz"
        )
        bound_stress = sp.sparse.load_npz(
            "./data/mech/" + str(idx_mu) + "/bound_stress_" + str(time) + ".npz"
        )

        u = np.load(
            "./data/mech/" + str(idx_mu) + "/displacement_" + str(time) + ".npy"
        )

        u_b_filled = np.load(
            "./data/mech/"
            + str(idx_mu)
            + "/displacement_boundary_"
            + str(time)
            + ".npy"
        )

        dim = 3
        fracture_faces_id = np.load(
            "./data/mech/" + str(idx_mu) + "/fracture_faces_id.npy"
        )
        normal_area =  np.load("./data/mech/" + str(idx_mu) + "/normal_area" + ".npy")
        face_area = np.load("./data/mech/" + str(idx_mu) + "/face_area" + ".npy")
        #normal = normal / faces_area
        normal = normal_area / np.linalg.norm(normal_area, ord=2)

        delta_u = u - u_ref
        delta_u_b_filled = u_b_filled - u_b_filled_ref

    
        # which one? # this one
        delta_T_area = stress_tensor_grad * delta_u + bound_stress * delta_u_b_filled
        delta_T_vect = np.reshape(delta_T_area, (dim, -1), order="F") / face_area
        delta_T_vect_frac = delta_T_vect[:, fracture_faces_id]

        T_area = delta_T_area 
        T_vect = delta_T_vect  
        T_vect_frac = delta_T_vect_frac  

        normal_projection = pp.map_geometry.normal_matrix(normal=normal)
        tangential_projetion = pp.map_geometry.tangent_matrix(normal=normal)

        T_normal = normal_projection @ T_vect_frac
        T_normal_normal = np.dot(normal, T_normal)
        T_normal_norm = np.linalg.norm(T_normal.T, ord=2, axis=1)
        T_tangential = tangential_projetion @ T_vect_frac
        T_tangential_y = T_tangential[1].T # one on-plane direction is aligned with y
        T_tangential_t = np.linalg.norm(np.array([T_tangential[0],T_tangential[2]]).T, ord=2, axis=1) # the other tangential dir, t, is made of x and z
        T_tangential_norm = np.linalg.norm(T_tangential.T, ord=2, axis=1)

        cff = T_tangential_norm - friction_coeff * T_normal_normal

        np.save(save_folder + "/cff_" + str(time), cff)
        
        exporter.write_vtu(
             [
                 (sd_fract, "T_xyz", T_vect_frac),
                 (sd_fract, "T_out_of_plane", T_normal_norm),
                 (sd_fract, "T_out_of_plane_signed", T_normal_normal),
                 (sd_fract, "minus_T_out_of_plane_signed", -T_normal_normal),
                 (sd_fract, "T_tangential_norm", T_tangential_norm),
                 (sd_fract, "T_tangential_y", T_tangential_y),
                 (sd_fract, "T_tangential_t", T_tangential_t),
                 (sd_fract, "cff", cff),
             ],
             time_dependent=True,
             time_step=time,
         )
         
    arguments = np.array_split(times_mech, 3)
    with ThreadPoolExecutor() as executor:
           output_executor = executor.map(parallel_fcn, arguments)

    # pay attention, executor doesnt print niether stops at errors! you have to force it printing the output

    for i in output_executor:
        print(i)

    print("Traction computed!\n")
