#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:45:01 2024

@author: ag19017
"""


#import skopt
import numpy as np
import pandas as pd
from math import log10, log
Ne = 3

a = log10(0.01367*2*10**(0.467) )
amin = a-0.10*a
amax = a+0.10*a


A = np.array([[amin,a,amax]])
#A
#dimensions = skopt.space.Space([(1.0,3.0),(0.6,1), (-2.33,-1.4)])
#lhsSampler = skopt.sampler.Lhs(criterion='maximin', iterations=1000)
#A=lhsSampler.generate(dimensions, Ne, random_state=7)
m = np.array(A).T
A = pd.DataFrame(m, columns = ['a'] )
A.to_excel('/scratch1/Diana/tesi_polito/Guendalina/Design_Matrix/DM_1',engine='xlsxwriter')
