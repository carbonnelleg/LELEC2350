# -*- coding: utf-8 -*-
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scikit-rf",
#     "scipy",
# ]
# ///
"""
Created on Wed May 04 16:57:44 2025

@author: carbonnelleg
"""
import skrf as rf
import scipy as sc
from scipy.constants import epsilon_0, mu_0
from scipy.integrate import trapezoid as trap
import numpy as np
from matplotlib import pyplot as plt