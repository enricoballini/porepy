import sys
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import time 
import pdb

os.system("clear")

n = 65
aaa_list = np.empty(n, dtype="object")
bbb_list = np.empty(n, dtype="object")
# aaa_list = np.zeros((n,100))

def fcn(i, q):
    """ """
    time.sleep(1)
    q.put((i, 0.77*i*np.ones(10)))

def fcn_2(args):
    """ """
    i = args[0]
    arg1 = args[1]
    time.sleep(1)
    print(arg1)
    print(i)
    return (i, 0.77*i*np.ones(100000))


# print("\n -------------------------------------------------------")
# queue.put is too slow

# q = mp.Queue()
# processes = []

# for i in np.arange(n):
#     print(i)
#     p = mp.Process(target=fcn, args=(i, q,))
#     processes.append(p)
#     p.start()

# for p in processes:
#     p.join()

# print("ran all processes")

# for i in np.arange(n):
#     # print(q.get_nowait())
#     # aaa_list[i] = q.get()
#     aaa_list[i], bbb_list[i] = q.get_nowait() 


# print("aaa_list = ", aaa_list)
# print("bbb_list = ", bbb_list)



print("\n -----------------------------------------------------")
arguments = []
bbb = []
for i in np.arange(n):
    arguments.append((i, "bla"))

with ProcessPoolExecutor() as executor:
    aaa = list(executor.map(fcn_2, arguments))
    ccc = executor.map(fcn_2, arguments)

for i in ccc:
    bbb.append(i)

print("\n aaa = ", aaa)
print("\n bbb = ", bbb)

print("\nDone!")