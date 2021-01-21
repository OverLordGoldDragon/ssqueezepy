# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 17:11:13 2021

@author: David
"""

import numpy as np
from scipy.signal import chirp,sweep_poly
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt
from ssqueezepy import ridge_extraction as rt

def viz(signal, tf_transf,ridge,flip_plot=False):
    plt.plot(signal)
    plt.title('Time signal')
    plt.xlabel('Time axis [A.U].')
    plt.ylabel('Signal Amplitude [A.U.]')
    plt.show()
    
    plt.figure()
    if(flip_plot):
        plt.imshow(np.flipud(np.abs(tf_transf)), aspect='auto', vmin=0, vmax=np.max(np.abs(tf_transf)), cmap='jet')
        plt.plot(len(Txo)-ridge,linestyle = '--',color='black')
        plt.title('Syncrosqueezed continouse wavelet transform w. ridge')
    else:
        plt.imshow(np.abs(tf_transf), aspect='auto', cmap='jet')
        plt.plot(ridge,linestyle = '--',color='black')
        plt.title('Continouse wavelet transform w. ridge')
   
    plt.xlabel('Time axis [A.U].')
    plt.ylabel('Frequency scales [A.U.]')
    plt.show()
    
    
### Example 0: example ridge from similar example as can be found at Mathworks website 'https://www.mathworks.com/help/wavelet/ref/wsstridge.html#bu6we25-penalty'    
test_matrix=np.array([[1, 4, 4],[2, 2, 2],[5,5,4]])
fs_test =np.exp(np.array([1,2,3]))

Energy,ridge_idx,_ = rt.extract_fridges(test_matrix,fs_test,penalty=2.0)

print('ridge follows indexes:', ridge_idx)
### Example 1: Two constant frequency signals
    
sig_lgth = 500
t_vec=np.linspace(0,10,sig_lgth,endpoint=True)

f1=0.5
f2=2.0
signal_1=np.sin(f1*2*np.pi*t_vec)
signal_2=np.cos(f2*2*np.pi*t_vec)

signal=signal_1+signal_2
padtype = 'wrap'

Txo, ssq_freq, Wxo, scales_xo, _ = ssq_cwt(signal,t=t_vec,padtype = padtype)
Wxo /= np.sqrt(scales_xo)  # L1 norm


penalty=20.0
# CWT example
Energy,ridge_idx,_ = rt.extract_fridges(Wxo,scales_xo,penalty,num_ridges=2,BW=25)
plt.figure()
viz(signal, Wxo, ridge_idx)

# syncrosqueezed example
max_Energy,ridge_idx,_ = rt.extract_fridges(Txo,scales_xo,penalty,num_ridges=2,BW=4)
plt.figure()
viz(signal, Txo, ridge_idx,flip_plot=True)


### Example 2: Two chirp signals with linear and quadratic frequrncy variation

sign_chirp_1 = chirp(t_vec, f0=2, f1=8, t1=20, method='linear')
sign_chirp_2 = chirp(t_vec, f0=.4, f1=4, t1=20, method='quadratic')

sign_chirp=sign_chirp_1+sign_chirp_2

padtype = 'reflect' # standard padding for ssq_cwt
Txo, ssq_freq, Wxo, scales_xo, _ = ssq_cwt(sign_chirp,t=t_vec,padtype = padtype)
Wxo /= np.sqrt(scales_xo)  # L1 norm

penalty=0.5

# CWT example
Energy,ridge_idx,_ = rt.extract_fridges(Wxo,scales_xo,penalty,num_ridges=2,BW=25)
plt.figure()
viz(sign_chirp, Wxo, ridge_idx)

# syncrosqueezed example
max_Energy,ridge_idx,_ = rt.extract_fridges(Txo,scales_xo,penalty,num_ridges=2,BW=2)
plt.figure()
viz(sign_chirp, Txo, ridge_idx,flip_plot=True)


# Example 3: Two sweep signals where respective frequency variations are described with polynomials


p1 = np.poly1d([0.025, -0.36, 1.25, 2.0])



sweep_sig_1 = sweep_poly(t_vec, p1)+signal_1


sweep_sig=sweep_sig_1
padtype = 'wrap'

Txo, ssq_freq, Wxo, scales_xo, _ = ssq_cwt(sweep_sig,t=t_vec,padtype = padtype)
Wxo /= np.sqrt(scales_xo)  # L1 norm

penalty=2.0

# CWT example
Energy,ridge_idx,_ = rt.extract_fridges(Wxo,scales_xo,penalty,num_ridges=2,BW=25)
plt.figure()
viz(sweep_sig, Wxo, ridge_idx)

# syncrosqueezed example
max_Energy,ridge_idx,_ = rt.extract_fridges(Txo,scales_xo,penalty,num_ridges=2,BW=2)
plt.figure()
viz(sweep_sig, Txo, ridge_idx,flip_plot=True)











