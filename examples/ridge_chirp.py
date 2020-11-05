import numpy as np
from numpy.fft import rfft

from ssqueezepy import ssq_cwt, issq_cwt
from ssqueezepy.toolkit import lin_band, cos_f, mad_rms
from ssqueezepy.viz_toolkit import imshow, plot, scat

#%%###########################################################################
def echirp(N):
    t = np.linspace(0, 10, N, False)
    return np.cos(2 * np.pi * np.exp(t / 3)), t
#%%## Configure signal #######################################################
N = 2048
noise_var = 6  # noise variance; compare error against = 10

x, ts = echirp(N)
x *= (1 + .3 * cos_f([1], N))  # amplitude modulation
xo = x.copy()
np.random.seed(4)
x += np.sqrt(noise_var) * np.random.randn(len(x))

#### Show signal & its global spectrum #######################################
axf = np.abs(rfft(x))

plot(xo); scat(xo, s=8, show=1)
plot(x);  scat(x,  s=8, show=1)
plot(axf, show=1)
#%%# Synchrosqueeze ##########################################################
wavelet = ('morlet', {'mu': 4.5})
Tx, fs, Wx, scales, w = ssq_cwt(x, wavelet, t=ts, nv=32, scales='log')
#%%# Visualize ###############################################################
pkw = dict(abs=1, w=.86, h=.9, aspect='auto', cmap='bone')
_Tx = np.pad(Tx, [[4, 4]])  # improve display of top- & bottom-most freqs
imshow(Wx, **pkw)
imshow(np.flipud(_Tx), norm=(0, 2e-1), **pkw)
#%%# Estimate inversion ridge ###############################################
bw, slope, offset = .035, .45, .45
Cs, freqband = lin_band(Tx, slope, offset, bw, norm=(0, 2e-1))
#%%###########################################################################
xrec = issq_cwt(Tx, wavelet, Cs, freqband)[0]
plot(xo)
plot(xrec, show=1)

axof   = np.abs(rfft(xo))
axrecf = np.abs(rfft(xrec))
plot(axof)
plot(axrecf, show=1)

print("signal   MAD/RMS: %.6f" % mad_rms(xo, xrec))
print("spectrum MAD/RMS: %.6f" % mad_rms(axof, axrecf))
