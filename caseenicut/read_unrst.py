import os
import numpy as np
import pdb

from ecl import EclipseFile

os.system("clear")



def read_simulated_summary( varsource, filename, init_date, singleread = True):
    """
    The function read the dates and the corresponding values of a specified 
    simulator variable output (summary)

    Args:
        varsource (string)      : Variable name and source name to read 
                                  with format <varname>:<sourcename>.
                                  Example 'WOPR:P4'
        filename                : Rootpath for thr reference data file to be
                                  read
        init_date               : Initial date for summary 
                                  TODO check if this argument can be eliminated
        singleread (True/False) : If true (default) only the variable for
                                  the source specified in varsource is 
                                  read. For full summary reading set false.
                            
                           
    Returns:
        dates: (list)       : List of datetime objects with the dates of report
                              steps where the simulator variable are given.
        prop: (numpy.array) : Array with the values of simulator output 
                              variable for the report steps 
                              (singleread = true). If all variables are
                              read, prop is a 2D array with all  variables
                              for all sources (size Number of timesteps x
                              Number of KW and WG)
        kw (numpy.array)    : Array with all the names of the keywords in
                              summary.
        wg (numpy.array)    : Array with all the names of well/groups in
                              summary.
    
    """
    # Read the SMSPEC file with KW names (variables) of WG names (sources)
    with EclipseFile (filename, 'SMSPEC') as store:
        wg = store.get('WGNAMES') # eclipse run
        if wg is None:
            wg = store.get('NAMES') #IX run
        kw = store.get('KEYWORDS')
    
    kw_upper = np.array([k.upper() for k in kw])
    wg_upper = np.array([w.upper() for w in wg])
    if 'DAY' in kw_upper:
        iday   = np.where(kw_upper=='DAY')
        imonth = np.where(kw_upper=='MONTH')
        iyear  = np.where(kw_upper=='YEAR')
    elif 'TIME' in kw_upper:
        itime  = np.where(kw_upper=='TIME')
    
    # Check if a specific variable or all variables have to be read
    if( singleread ):
        prop_elem   = varsource.upper().split(':')
        iprop       = np.where(kw_upper==prop_elem[0])
        Nvarsources = 1
        if len(prop_elem)>1:        
            iwell = np.where(wg_upper==prop_elem[1].upper())
            iprop = iprop[0][np.array([elem in iwell[0] for elem in iprop[0]])]
            iprop = iprop[-1]
        else:
            iprop = iprop[-1]
    else:
        Nvarsources = len(kw)

    with EclipseFile (filename, 'UNSMRY') as store:
        # Get the number of steps
        total = store.cnt['PARAMS  '] 
        prop = np.zeros((total,Nvarsources))
        dates = []
        for j in range(0,total):
            vec_dat = store.get(kwd='PARAMS  ', seq=j)
            if( singleread ) :
                prop[j,0] = vec_dat[iprop]
            else:
                prop[j,:] = vec_dat
            if ('iday' in locals()):
                day = int(vec_dat[iday])
                month = int(vec_dat[imonth])
                year = int(vec_dat[iyear])
                if day==0 and month==0 and year==0:
                    date = init_date
                else:
                    date = '%d/%d/%d' % (day, month, year)
                    date = datetime.strptime(date, '%d/%m/%Y')
            else:
                date = float(vec_dat[itime])
                date = init_date + timedelta(days=date)
            dates.extend([date])
    return dates, prop, kw_upper, wg_upper


def read_and_save_variables(data_folder, save_folder, unrst_file_name, idx_mu, variables: list[str]):
    """

    """
    os.system("mkdir " + save_folder + "/" + str(idx_mu))
    file_name = data_folder + "/" + str(idx_mu) + "/" + unrst_file_name

    with EclipseFile (file_name, 'SMSPEC') as store:
        wg = store.get('WGNAMES') 
        if wg is None:
            wg = store.get('NAMES')
        keywords = store.get('KEYWORDS')

    idx_keywords = [np.where(keywords == var)[0][0] for var in variables]

    with EclipseFile(file_name, "UNSMRY") as store:
        total = store.cnt["PARAMS  "]
        
        for j, var in enumerate(variables):
            tmp = 9999999*np.ones(total)

            for i in range(0, total):
                # time = i * timestep
                tmp[i] = store.get(kwd="PARAMS  ", seq=i)[idx_keywords][j]

            np.save(save_folder + "/" + str(idx_mu) + "/" + var, tmp)


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
    
    