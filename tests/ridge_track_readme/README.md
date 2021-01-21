# Frequency_ridge_tracking
Ridge tracking extracts the N (user selected integer) most prominent frequency ridges from a Time-frequency representation. 

The method is based on a forward-backward greedy algorithm that penalises frequency jumps similar to the MATLAB function 'tfridge' (https://de.mathworks.com/help/signal/ref/tfridge.html). 

Further information about algorithm as well as limitations and comparisson to other ridge extraction schemes can be found in publication [2] (below):



### Examples 

## 1: Two constant frequency signals

```python

    
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




```

![signal_1](/tests/ridge_track_readme/imgs/signal_1.png)
![cwt_ridge_signal_1](/tests/ridge_track_readme/imgs/cwt_signal_1_ridge.png)
![ssq_ridge_signal_1](/tests/ridge_track_readme/imgs/ssq_signal_1_ridge.png)

## 2: Two chirp signals with linear and quadratic frequency variation

```python

    
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



```

![signal_2](/tests/ridge_track_readme/imgs/signal_2.png)
![cwt_ridge_signal_2](/tests/ridge_track_readme/imgs/cwt_signal_2_ridge.png)
![ssq_ridge_signal_2](/tests/ridge_track_readme/imgs/ssq_signal_2_ridge.png)

## 3: One sweep signal and one constant frequency signal

```python

    
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



```

![signal_3](/tests/ridge_track_readme/imgs/signal_3.png)
![cwt_ridge_signal_3](/tests/ridge_track_readme/imgs/cwt_signal_3_ridge.png)
![ssq_ridge_signal_3](/tests/ridge_track_readme/imgs/ssq_signal_3_ridge.png)

### Q&A

## 1. signal padding

Particular consideration to choice of wavelet type, signal padding and other input parameters are advised before using the time-frequency ridge extraction. 
Different padding schems suited to your time signals may be nessecary to handle edge effects in your time frequency maps (see example 1 and 3).  

## 2. ridge extraction on ssq_cwt

The idea of a syncrosqueezed time-frequency representation is to improve upon the localization process and more accurately recover the frequency components compared to inverting the CWT over the entire time-scale plan [1]. However, whether ridge extraction is improved on a syncrosqueezed transform is still uncertain and may require further considerations [2] and expertise from the users on appropriate parameter tuning for stable and improved results.  

[1] The Synchrosqueezing algorithm for time-varying spectral analysis: robustness properties and new paleoclimate applications. Gaurav Thakur, Eugene Brevdo, Neven S. Fuˇckar, and Hau-Tieng Wu. https://arxiv.org/pdf/1105.0010.pdf <br>
[2] On the extraction of instantaneous frequencies fromridges in time-frequency representations of signals. D. Iatsenko, P. V. E. McClintock, A. Stefanovska. https://arxiv.org/pdf/1310.7276.pdf