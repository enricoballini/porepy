import os
import sys

import numpy as np
import multiprocessing as mp


print("inside bbbb.py")

def fcnpp():
    print("inside fcnpp")

def generate_snapshots(
):
    """
    """
    
    index_list = np.arange(0,4)
    n_proc = 2

    def one_step(
    ): 
        """ """
        print(
            "process ",
            os.getpid(),
            " started. It will genarete snapshots number "
        )

        for i in np.arange(index_list.shape[0]):
            print("going to run snap number: ", index_list[i])
            print("finished to run snap number: ", index_list[i])

    
    proc_list = [None] * n_proc
    for ii, proc_id in enumerate(np.arange(n_proc)):
        proc_list[ii] = mp.Process(
            target=one_step,
            args=(),
        )
        proc_list[ii].start()
        
    for proc in proc_list:
        proc.join()