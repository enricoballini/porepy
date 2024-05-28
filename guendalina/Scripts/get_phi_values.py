#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:08:28 2024

@author: ag19017
"""

import numpy as np
from math import log10, log

# from Table page 10 report: stress and lambda values for each reservoir region (total: 6 regions)
lam = np.array([0.040429084, 0.038882959, 0.03935942, 0.041984826, 0.03978687, 0.036866925])
stress = np.array([315.18,316.67,317.44,319.01,321.20,325.36])

b=-1.16434
a = log10(0.01367*2*10**(0.467) )#-1.

cm = 10**(a+b*np.log10(stress))

e0 = lam/(stress*cm)-1
phi=e0/(1+e0)
