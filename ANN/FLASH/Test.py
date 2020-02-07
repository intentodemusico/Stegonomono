# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 20:34:37 2019

@author: INTENTODEMUSICO
"""

import pyrem as pr
import numpy as np

noise = np.random.normal(size=int(1e4))
activity, complexity, morbidity = pr.univariate.hjorth(noise)