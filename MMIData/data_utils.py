from .global_imports import *

def moving_average(a, n=5):
    return np.convolve(a, np.ones(n), 'valid') / n

def MA_zeropad(a, n=5):
	z = np.zeros_like(a)
	z[n-1:] = moving_average(a, n)
	return z

