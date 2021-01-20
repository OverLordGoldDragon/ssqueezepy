# Frequency_ridge_tracking
Ridge tracking extracts the N (user selected integer) most prominent frequency ridges from a Time-frequency representation. 

The method is based on a forward-backward greedy algorithm that penalises frequency jumps similar to the MATLAB function 'tfridge' (https://de.mathworks.com/help/signal/ref/tfridge.html). 

Further information about algorithm as well as limitations and comparisson to other ridge extraction schemes can be found in the following publication:
 'On the extraction of instantaneous frequencies fromridges in time-frequency representations of signals",D. Iatsenko, P. V. E. McClintock, A. Stefanovska, https://arxiv.org/pdf/1310.7276.pdf



### Example 1: Two constant frequency signals

```python

    
sig_lgth = 500
t_vec=np.linspace(0,10,sig_lgth)

f1=0.5
f2=2.0
signal_1=np.sin(f1*2*np.pi*t_vec)
signal_2=np.cos(f2*2*np.pi*t_vec)

signal=signal_1+signal_2

Txo, ssq_freq, Wxo, scales_xo, _ = ssq_cwt(signal,t=t_vec)
Wxo /= np.sqrt(scales_xo)  # L1 norm


penalty=2.0
# CWT example
Energy,ridge_idx,_ = rt.extract_fridges(Wxo,scales_xo,penalty,num_ridges=2,BW=25)
plt.figure()
viz(signal, Wxo, ridge_idx)

# syncrosqueezed example
max_Energy,ridge_idx,_ = rt.extract_fridges(Txo,scales_xo,penalty,num_ridges=2,BW=2)
plt.figure()
viz(signal, Txo, ridge_idx,flip_plot=True)




```

![signal_1](/imgs/signal_1.png)
![cwt_ridge_signal_1](/imgs/cwt_signal_1_ridge.png)
![ssq_ridge_signal_1](/imgs/ssq_signal_1_ridge.png)

### Example 2: Two chirp signals with linear and quadratic frequency variation

```python

    
sign_chirp_1 = chirp(t_vec, f0=2, f1=8, t1=20, method='linear')
sign_chirp_2 = chirp(t_vec, f0=.4, f1=4, t1=20, method='quadratic')

sign_chirp=sign_chirp_1+sign_chirp_2


Txo, ssq_freq, Wxo, scales_xo, _ = ssq_cwt(sign_chirp, wavelet=('morlet'),t=t_vec)
Wxo /= np.sqrt(scales_xo)  # L1 norm

penalty=.3

# CWT example
Energy,ridge_idx,_ = rt.extract_fridges(Wxo,scales_xo,penalty,num_ridges=2,BW=25)
plt.figure()
viz(sign_chirp, Wxo, ridge_idx)

# syncrosqueezed example
max_Energy,ridge_idx,_ = rt.extract_fridges(Txo,scales_xo,penalty,num_ridges=2,BW=2)
plt.figure()
viz(sign_chirp, Txo, ridge_idx,flip_plot=True)



```

![signal_2](/imgs/signal_2.png)
![cwt_ridge_signal_2](/imgs/cwt_signal_2_ridge.png)
![ssq_ridge_signal_2](/imgs/ssq_signal_2_ridge.png)

### Example 3: One sweep signal and one constant frequency signal

```python

    
p1 = np.poly1d([0.025, -0.36, 1.25, 2.0])



sweep_sig_1 = sweep_poly(t_vec, p1)+signal_1


sweep_sig=sweep_sig_1

Txo, ssq_freq, Wxo, scales_xo, _ = ssq_cwt(sweep_sig, wavelet=('morlet'),t=t_vec)
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



```

![signal_3](/imgs/signal_3.png)
![cwt_ridge_signal_3](/imgs/cwt_signal_3_ridge.png)
![ssq_ridge_signal_3](/imgs/ssq_signal_3_ridge.png)
