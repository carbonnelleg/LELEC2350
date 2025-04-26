# -*- coding: utf-8 -*-
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scikit-rf",
# ]
# ///
"""
Created on Sat Apr 26 08:42:35 2025

@author: carbonnelleg
"""

import skrf as rf
import numpy as np
from matplotlib import pyplot as plt
"""
_______________________________________________________________________________
Importing VNA data
"""
x_offset = np.arange(0, 70, 5)
ref_ntw = rf.Network(__file__ + '/../no_arago.s2p')
arago_ntws = [rf.Network(__file__ + f'/../arago_{x}.s2p')/ref_ntw for x in x_offset]

mean_pow = [np.mean(np.abs(ntw.s[:,1,0])) for ntw in arago_ntws]
f_11Ghz_pow = [ntw.s[ntw.f.size//2,1,0] for ntw in arago_ntws]

plt.figure()
plt.plot(x_offset, 20*np.log10(mean_pow), label='Average over all frequencies')
plt.plot(x_offset, 20*np.log10(np.abs(f_11Ghz_pow)), label='Power at f = 11 GHz')
plt.legend()

plt.figure()
for i, ntw in enumerate(arago_ntws[5:8], start=5):
    ntw.name = f'Offset of {x_offset[i]} cm'
    ntw.plot_s_db(m=0, n=1)
plt.show()
