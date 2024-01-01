from obspy import read
import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
from math import sin, cos
from obspy import Trace

def zero_crossing(trace, slop=2):
    '''
    Description {}

    :type trace: obspy.core.trace.Trace
    :param trace:
    :type slop: int
    :param slop:
         2: indicate zero crossing with positive slop
        -2: indicate zero crossing with negetive slop
    :return: a NumPy array of zero crossing positions in seconds unit.
    '''
    data = trace.data
    sps = trace.stats.sampling_rate
    ###
    sings = np.sign(data)
    sings_changing = np.diff(sings)
    sings_changing_location = np.where(sings_changing==slop)[0]
    return sings_changing_location / sps
