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
    sings_changing_location = np.where(sings_changing == slop)[0]
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
    if integral2 < 0:
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


def FilterChebyshev(st, f, dfpar, fstart, cycles):
    '''
    Docstring {}
    '''
    st2 = st.copy()
    tr_v = st2.select(channel='*[Z1]')[0]
    tr_n = st2.select(channel='*[N2]')[0]
    tr_e = st2.select(channel='*[E3]')[0]
    #
    DT = cycles / f
    wl = round(DT/tr_v.stats.delta)
    fnyq = tr_v.stats.sampling_rate / 2
    df = dfpar * f
    fmin = max(fstart, f-df/2)
    fmax = min(fnyq, f+df/2)
    #
    wp = np.array([fmin+(fmax-fmin)/20, fmax-(fmax-fmin)/20]) / fnyq
    ws = np.array([fmin-(fmax-fmin)/20, fmax+(fmax-fmin)/20]) / fnyq
    #
    na, wn = scipy.signal.cheb1ord(
        wp=wp, ws=ws, gpass=1, gstop=5, analog=False, fs=None)
    ch1, ch2 = scipy.signal.cheby1(na, 0.5, wn, btype='bandpass')
    #
    norths = scipy.signal.lfilter(ch1, ch2, tr_n.data)
    easts = scipy.signal.lfilter(ch1, ch2, tr_e.data)
    verts = scipy.signal.lfilter(ch1, ch2, tr_v.data)
    #
    tr_v.data = verts
    tr_n.data = norths
    tr_e.data = easts
    return st2


def Shifting_HvsV(freq, sps):
    '''
    Docstring {}
    '''
    delta = 1 / sps
    shift_sample = math.floor(1/(4*freq*delta))
    shift_second = shift_sample / sps
    return shift_sample, shift_second


def Windowing(st, length, freq):
    '''
    Docstring {}
    '''
    tr_v = st.select(channel='*[Z1]')[0]
    tr_n = st.select(channel='*[N2]')[0]
    tr_e = st.select(channel='*[E3]')[0]
    ###
    tr_time_duration = tr_v.stats.npts / tr_v.stats.sampling_rate
    zcrossing = ZeroCrossing(trace=tr_v, slop=2)
    zcrossing = zcrossing[
        np.where(zcrossing <= tr_time_duration-length)
        ]
    _, shift_second = Shifting_HvsV(freq, tr_v.stats.sampling_rate)
    ###
    lst_tr_v = []
    lst_tr_n = []
    lst_tr_e = []
    for stime in zcrossing:
        stime = tr_v.stats.starttime + stime
        v_trim = tr_v.slice(starttime=stime, endtime=stime+length)
        stime = stime - shift_second
        n_trim = tr_n.slice(starttime=stime, endtime=stime+length)
        e_trim = tr_e.slice(starttime=stime, endtime=stime+length)
        if e_trim.stats.npts != v_trim.stats.npts:
            break
        ###
        lst_tr_v.append(v_trim)
        lst_tr_n.append(n_trim)
        lst_tr_e.append(e_trim)
    return lst_tr_v, lst_tr_n, lst_tr_e
