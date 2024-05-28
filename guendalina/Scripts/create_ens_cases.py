#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 08:45:25 2023

@author: ext2035231
"""
import numpy as np
import os
import shutil
from scipy import io
from math import log10, log
import random
import pandas as pd

# Load or create design Matrix
a = log10(0.01367*2*10**(0.467) )
amin = a-0.10*a
amax = a+0.10*a

A = np.array([[amin,a,amax]]).T

Ne = A.shape[0]

Nl = 27 #layers
Nr = 6 #rocktab
Nunder = 10
Nover = 10

# Valore di porosità per i vari layers, calclolato a ritroso partendo da lambda nel file gmt 
phi = np.array([0.22999866, 0.20000097, 0.21000063, 0.2600013 , 0.21999885, 0.15999926])
# Valori degli stress da report
under_stress = np.array([390.98,453.12,516.13,579.90,644.36,709.47,775.16,841.39,908.14,975.36])
stress = np.array([315.18,316.67,317.44,319.01,321.20,325.36])
over_stress = np.array([11.54,40.29,71.76,104.83,139.05,174.18,210.06,246.58,283.65,321.22])

# Valori da relazione dichiarata nel report
e0 = phi/(1-phi)
b=-1.16434

# Calcolo cm per tutti gli elementi dell'ensemble
cm_under_ens = 10**(A+b*np.log10(under_stress))
cm_ens = 10**(A+b*np.log10(stress))
cm_over_ens = 10**(A+b*np.log10(over_stress))

# Calcolo lambda
lamb_ens =cm_ens*stress*(1+e0)

# Creo l'input per il gmt
#Copio il file parametrizzato in tutte le cartelle e lo modifico
src2 = '/scratch1/Diana/tesi_polito/Guendalina/BC_FILES/GUE2022_TESI.data'

for i in range(1,Ne+1):
    cm_under=cm_under_ens[i-1,:]
    cm = cm_ens[i-1,:]
    cm_over = cm_over_ens[i-1,:]
    lamb = lamb_ens[i-1,:]
    dest_fpath='/scratch1/Diana/tesi_polito/Guendalina/Ensemble1/RUN_'+str(i)
    os.makedirs(dest_fpath, exist_ok=True)
    shutil.copy2(src2, dest_fpath)

    #Sostituisco i parametri nel file copiato
    with open('/scratch1/Diana/tesi_polito/Guendalina/Ensemble1/RUN_'+str(i)+'/GUE2022_TESI.data','r') as file:
        filedata = file.read()
        for rr in range(1,Nr+1):
            #valori di cm in giacimento
            filedata = filedata.replace('__CMR'+str(rr)+'__', str(np.format_float_scientific(cm[rr-1],precision=2)))
            filedata = filedata.replace('__LMR'+str(rr)+'__', str(lamb[rr-1].round(8)))
        for tt in range(1,Nunder+1):
            #valori di cm under-burden
            filedata = filedata.replace('__CMU'+str(tt)+'__', str(np.format_float_scientific(cm_under[tt-1],precision=2)))
        for tt in range(1,Nover+1):
            #valori di cm over-burden
            filedata = filedata.replace('__CMO'+str(tt)+'__', str(np.format_float_scientific(cm_over[tt-1],precision=2)))
    #Salvo il file
    with open('/scratch1/Diana/tesi_polito/Guendalina/Ensemble1/RUN_'+str(i)+'/GUE2022_TESI.data','wt') as file:
        file.write(filedata)

# Creo l'input per Eclipse


# creo le tabelle ROCKTAB
kappa_ens = lamb_ens/3

# Valori degli intervalli di pressione dal file .DATA del caso base
sigma_t = np.array([[716.1916683,551.6374108,455.3793409,387.0831534,334.1085148,290.8250835,254.2294638,222.528896,194.5670136,169.5542574,146.9274672,126.2708261,107.2685583,89.67520634,73.29618749,57.9746386,43.5822557,30.01275617,17.17711073,5],	    
[720.9044582,555.2597639,458.3638293,389.6150696,336.2893887,292.7191349,255.8810103,223.9703752,195.8232003,170.6446943,147.8679652,127.0744406,107.9462522,90.23631593,73.74875971,58.32568088,43.83792527,30.17850595,17.25780369,5],
[723.5123146,557.2642191,460.0153174,391.0161236,337.496191,293.767222,256.7949055,224.7680281,196.5183203,171.2480955,148.3883965,127.5191265,108.321259,90.54680997,73.99919381,58.51993266,43.97940188,30.27022479,17.30245573,5],
[726.2712272,559.3847795,461.7624656,392.4983317,338.7728955,294.8760179,257.7617367,225.6118839,197.2537041,171.8864478,148.9389731,127.9895701,108.7179875,90.87528893,74.26413394,58.72543619,44.1290733,30.36725631,17.34969417,5],
[731.0631693,563.0679705,464.7970789,395.0727718,340.9903975,296.8018802,259.4410212,227.077573,198.5309886,172.9951987,149.8952669,128.8066815,109.4070634,91.44582244,74.72430718,59.08237428,44.38903688,30.53578989,17.43174239,5],
[739.5979032,569.627953,470.2019059,399.6580028,344.9399005,300.2319556,262.4319268,229.6880525,200.8059085,174.9699502,151.5984832,130.2620054,110.6343485,92.46197658,75.5439031,59.71810227,44.85204749,30.83595823,17.57787515,5]]).T

n = 20 #numero degli intervalli di pressione


# Copio il file parametrizzato
src2 = '/scratch1/Diana/tesi_polito/Guendalina/BC_FILES/GUE2022_UPP2_TESI.DATA'
for i in range(1,Ne+1):
    shutil.copy2(src2, '/scratch1/Diana/tesi_polito/Guendalina/Ensemble1/RUN_'+str(i))    
    e1 = np.zeros((n,Nr))
    v = np.zeros(n)
    kappa = kappa_ens[i-1,:]
    lamb = lamb_ens[i-1,:]
    f = open('/scratch1/Diana/tesi_polito/Guendalina/Ensemble1/RUN_'+str(i)+'/ROCKTABH.INC','w')
    f.write('ROCKTABH\n')
    for k in range(1,Nr+1):
        f.write('--Regione' + str(k))
        f.write('lambda = ' +str(lamb[k-1])+'\n')
        for j in range(1,n+1):
            v[j-1] = np.log(sigma_t[j-1,k-1]/stress[k-1])
            #linea di normal-compressione
            e1[j-1,k-1]=e0[k-1]-lamb[k-1]*np.log(sigma_t[j-1,k-1]/stress[k-1]) #trovo e_N
        for j in range(1,n):
            sigma1 = sigma_t[:,k-1]
            #linee di scarico-ricarico
            ek1=e1[j-1,k-1]-kappa[k-1]*np.log(sigma1/sigma_t[j-1,k-1])
            M1=ek1/e0[k-1]
            rm = M1.size
            Table2 = np.array([sigma1[j-1:20].round(2),M1[j-1:20].round(8),np.ones(rm-j+1)])
            Table2 = Table2.T
            if (rm>1):
                 for x in range(rm-j+1):
                      f.write(str(Table2[x,0])+'                                     '+str(Table2[x,1])+'                           '+str(Table2[x,2])+'\n')
                 f.write('/'+'\n')
        f.write('/'+'\n')
        
    f.close()
    

