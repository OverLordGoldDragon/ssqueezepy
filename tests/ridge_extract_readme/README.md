# Ridge extraction

Extracts the `n_ridges` (user selected integer) most prominent frequency ridges from a time-frequency representation. 

The method is based on a forward-backward greedy path optimization algorithm that penalises frequency jumps similar to the MATLAB function 'tfridge' (https://de.mathworks.com/help/signal/ref/tfridge.html). 

Further information about the particular algorithm (version of eq. III.4 in publication) as well as limitations and comparisson to other ridge extraction schemes can be found in publication [2] (below):



### Examples 

## 1: Two constant frequency signals (ssq_cwt instability)

```python   
"""Sine + cosine."""
N, f1, f2 = 500, 0.5, 2.0
padtype = 'wrap'
penalty = 20

t = np.linspace(0, 10, N, endpoint=True)
x1 = np.sin(2*np.pi * f1 * t)
x2 = np.cos(2*np.pi * f2 * t)
x = x1 + x2

Tx, ssq_freqs, Wx, scales = ssq_cwt(x, t=t, padtype=padtype)

# CWT example
ridge_idxs = extract_ridges(Wx, scales, penalty, n_ridges=2, bw=25)
viz(x, Wx, ridge_idxs, scales)

# SSQ_CWT example
ridge_idxs = extract_ridges(Tx, scales, penalty, n_ridges=2, bw=4)
viz(x, Tx, ridge_idxs, ssq_freqs, ssq=True)
```

![signal_1](/tests/ridge_extract_readme/imgs/signal_1.png)
![cwt_ridge_signal_1](/tests/ridge_extract_readme/imgs/cwt_signal_1_ridge.png)
![ssq_ridge_signal_1](/tests/ridge_extract_readme/imgs/ssq_signal_1_ridge.png)

## 2: Two chirp signals with linear and quadratic frequency variation

```python
"""Linear + quadratic chirp."""
N = 500
penalty = 0.5
padtype = 'reflect'

t = np.linspace(0, 10, N, endpoint=True)
x1 = sig.chirp(t, f0=2,  f1=8, t1=20, method='linear')
x2 = sig.chirp(t, f0=.4, f1=4, t1=20, method='quadratic')
x = x1 + x2

Tx, ssq_freq, Wx, scales = ssq_cwt(x, t=t, padtype=padtype)

# CWT example
ridge_idxs = extract_ridges(Wx, scales, penalty, n_ridges=2, bw=25)
viz(x, Wx, ridge_idxs)

# SSQ_CWT example
ridge_idxs = extract_ridges(Tx, scales, penalty, n_ridges=2, bw=2)
viz(x, Tx, ridge_idxs, ssq=True)
```

![signal_2](/tests/ridge_extract_readme/imgs/signal_2.png)
![cwt_ridge_signal_2](/tests/ridge_extract_readme/imgs/cwt_signal_2_ridge.png)
![ssq_ridge_signal_2](/tests/ridge_extract_readme/imgs/ssq_signal_2_ridge.png)

## 3: One sweep signal and one constant frequency signal

```python    
"""Cubic polynomial frequency variation + pure tone."""
N, f = 500, 0.5
padtype = 'wrap'

t = np.linspace(0, 10, N, endpoint=True)
p1 = np.poly1d([0.025, -0.36, 1.25, 2.0])
x1 = sig.sweep_poly(t, p1)
x2 = np.sin(2*np.pi * f * t)
x = x1 + x2

Tx, ssq_freq, Wx, scales = ssq_cwt(x, t=t, padtype=padtype)

# CWT example
penalty = 2.0
ridge_idxs = extract_ridges(Wx, scales, penalty, n_ridges=2, bw=25)
viz(x, Wx, ridge_idxs)

# SSQ_CWT example
ridge_idxs = extract_ridges(Tx, scales, penalty, n_ridges=2, bw=2)
viz(x, Tx, ridge_idxs, ssq=True)
```

![signal_3](/tests/ridge_extract_readme/imgs/signal_3.png)
![cwt_ridge_signal_3](/tests/ridge_extract_readme/imgs/cwt_signal_3_ridge.png)
![ssq_ridge_signal_3](/tests/ridge_extract_readme/imgs/ssq_signal_3_ridge.png)

## 4: Example of insufficient penalty term 


```python
"""Linear + quadratic chirp."""
N = 600
padtype = 'symmetric'
t = np.linspace(0, 3, N, endpoint=True)

x1 = sig.chirp(t-1.5, f0=30, t1=1.1, f1=40, method='quadratic')
x2 = sig.chirp(t-1.5, f0=10, t1=1.1, f1=5,  method='quadratic')
x = x1 + x2

Tx, ssq_freq, Wx, scales = ssq_cwt(x, t=t, padtype=padtype)

# CWT example no penalty
ridge_idxs = extract_ridges(Wx, scales, penalty=0.0, n_ridges=2, bw=25)
viz(x, Wx, ridge_idxs)

# CWT example with penalty
ridge_idxs = extract_ridges(Wx, scales, penalty=0.5, n_ridges=2, bw=25)
viz(x, Wx, ridge_idxs)
```

![signal_failed_wsst](/tests/ridge_extract_readme/imgs/signal_failed_wsst.png)
![cwt_ridge_signal_failed_wsst_noPen](/tests/ridge_extract_readme/imgs/cwt_signal_failed_wsst_ridge_pen00.png)
![cwt_ridge_signal_failed_wsst_penalty05](/tests/ridge_extract_readme/imgs/cwt_signal_failed_wsst_ridge_pen05.png)


### Practical considerations

## 1. Ridge extraction on ssq_cwt

The idea of a syncrosqueezed time-frequency representation is to improve upon the localization process and more accurately recover the frequency components compared to inverting the CWT over the entire time-scale plane [2]. However, whether ridge extraction is improved on a syncrosqueezed transform is still uncertain and may require further considerations [1] and expertise from the users on appropriate parameter tuning for stable and improved results.  


[1] On the extraction of instantaneous frequencies fromridges in time-frequency representations of signals. D. Iatsenko, P. V. E. McClintock, A. Stefanovska. https://arxiv.org/pdf/1310.7276.pdf <br>
[2] The Synchrosqueezing algorithm for time-varying spectral analysis: robustness properties and new paleoclimate applications. Gaurav Thakur, Eugene Brevdo, Neven S. Fuˇckar, and Hau-Tieng Wu. https://arxiv.org/pdf/1105.0010.pdf

Example 1 particularly highlights how edge-effects may make ridge extraction on the syncrosqueezed transform more unstable. 

## 2. Signal padding

Particular consideration to choice of wavelet type, signal padding and other input parameters are advised before using the time-frequency ridge extraction. 
Different padding schems suited to your time signals may be nessecary to handle edge effects in your time frequency maps. Example 2 & 3 highlights an appropriate padding along with usage of wavelet.


## 3. Penalty term

Example 4 highlights the effect of the increased penalty term to deter frequency jumps by the ridge extraction.
