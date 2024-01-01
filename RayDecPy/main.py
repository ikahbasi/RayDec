from obspy import read
import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
from math import sin, cos
from obspy import Trace

def ZeroCrossing(trace, slop=2):
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


def WindowLength(freq, sps, cycles=10):
    '''
    Description {}
    '''
    delta = 1 / sps
    seconds = cycles / freq
    samples = seconds / delta
    return seconds, int(samples)


def Theta(tr_v, tr_e, tr_n):
    '''
    EQ.4
    Description {}
    '''
    integral1 = sum(tr_v.data*tr_e.data)
    integral2 = sum(tr_v.data*tr_n.data)
    theta = np.arctan(integral1/integral2)   
    if integral2<0:
        theta = theta + np.pi
    theta = np.mod(theta+np.pi, 2*np.pi)
    return theta


def NormalizedCorrelation(sig1, sig2):
    '''
    EQ.5
    Description {}
    '''
    numerator = sum(sig1*sig2)
    denominator = np.sqrt(sum(sig1*sig1)*sum(sig2*sig2))
    corr = numerator / denominator
    return corr