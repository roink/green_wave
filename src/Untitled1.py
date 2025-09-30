#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# In[15]:


doy_values = [  1,  17,  33,  49,  65,  81,  97, 113, 129 ,145 ,161, 177 ,193, 209, 225, 241 ,257 ,273, 289 ,305 ,321, 337 ,353]
ndvi_values = [5809., 5809., 2657.,2657.,2657.,2780.,6257.,6346.,6759.,7396.,8015.,8058., 8015.,8058.,7991.,8019.,7991.,7410.,7286.,6416.,5859.,5859.,1407.]


# In[58]:


def double_logistic(t, xmidSNDVI, scalSNDVI, xmidANDVI, scalANDVI, bias, scale):
    """
    Double-logistic function for fitting NDVI profiles.
    """
    spring = 1 / (1 + np.exp((xmidSNDVI - t) / scalSNDVI))
    autumn = 1 / (1 + np.exp((xmidANDVI - t) / scalANDVI))
    return bias + scale * (spring - autumn)


# In[59]:


def double_logistic(t, xmidSNDVI, bias, scale):
    
    return bias + scale * np.sin((t-xmidSNDVI)/182.5)
    


# In[ ]:


def tanh_sin(t, xmidSNDVI, bias, scale, k):
    
    return bias + scale * np.tanh(np.sin((t-xmidSNDVI)/365) *k) / np.tanh(k)
    


# In[60]:


# Define the time range (DOY 0-365)
t_values = np.linspace(0, 365, 366)

# Define initial guesses
bias_guess = 1000  # Example bias
scale_guess = 5000  # Example scale
initial_guess = [120, 20, 270 ]  # Northern hemisphere


# In[61]:


# Fit the double-logistic function
params, _ = curve_fit(double_logistic, doy_values, ndvi_values, p0=initial_guess)

# Generate fitted curve
doy_full = np.arange(1, 366)  # Full year for smooth plotting
ndvi_fitted = double_logistic(doy_full, *params)

# Plot fitted curve
plt.figure(figsize=(10, 5))
plt.scatter(doy_values, ndvi_values, color="black", label="Observed NDVI")
plt.plot(doy_full, ndvi_fitted, color="red", linestyle="--", label="Fitted Double-Logistic Curve")


plt.xlabel("Day of Year")
plt.ylabel("NDVI")
plt.title(f"NDVI Seasonal Curve Fitting" )
plt.legend()
plt.grid(True)
plt.show()


# In[79]:


def tanh_sin(t, xmidSNDVI, bias, scale, k):
    
    return bias + scale * np.tanh(np.sin((t-xmidSNDVI)*2*np.pi/365) *k*2*np.pi/365) / np.tanh(k*2*np.pi/365)
    


# In[82]:


# Define initial guesses
bias_guess = 1000  # Example bias
scale_guess = 5000  # Example scale
initial_guess = [120, 1000, 5000,8 ]  # Northern hemisphere


# In[83]:


# Fit the double-logistic function
params, _ = curve_fit(tanh_sin, doy_values, ndvi_values, p0=initial_guess)

# Generate fitted curve
doy_full = np.arange(1, 366)  # Full year for smooth plotting
ndvi_fitted = tanh_sin(doy_full, *params)

# Plot fitted curve
plt.figure(figsize=(10, 5))
plt.scatter(doy_values, ndvi_values, color="black", label="Observed NDVI")
plt.plot(doy_full, ndvi_fitted, color="red", linestyle="--", label="Fitted Double-Logistic Curve")


plt.xlabel("Day of Year")
plt.ylabel("NDVI")
plt.title(f"NDVI Seasonal Curve Fitting" )
plt.legend()
plt.grid(True)
plt.show()


# In[88]:


import numpy as np
from scipy.optimize import curve_fit

def sigmoid_sine(x, a, k):
    """
    Sigmoid-sine composition with sharpened transitions.
    
    Args:
        x    : Input array (phase values in radians)
        a    : Amplitude parameter
        k    : Steepness parameter (higher = sharper transitions)
    
    Returns:
        y    : Transformed waveform
    """
    numerator = 2 / (1 + np.exp(-k  * np.sin(x))) - 1  # Logistic-sine term
    return a * numerator 

# Example usage with synthetic data
if True:
    # Generate noisy test data
    x_data = np.linspace(0, 4*np.pi, 200)
    true_a = 2.5
    true_k = 4.0
    y_data = sigmoid_sine(x_data, true_a, true_k) + 0.1 * np.random.normal(size=len(x_data))

    # Perform curve fitting
    params_guess = [1.0, 1.0]  # Initial guess for [a, k]
    popt, pcov = curve_fit(sigmoid_sine, x_data, y_data, p0=params_guess)

    # Extract fitted parameters
    a_fit, k_fit = popt
    print(f"True parameters: a={true_a}, k={true_k}")
    print(f"Fitted parameters: a={a_fit:.3f}, k={k_fit:.3f}")


# In[89]:


plt.figure(figsize=(10, 4))
plt.plot(x_data, y_data, 'b.', label='Noisy data')
plt.plot(x_data, sigmoid_sine(x_data, *popt), 'r-', label='Fit')
plt.xlabel('Phase (radians)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()


# In[143]:


def double_sigmoid_sine(t, xmidSNDVI, scalSNDVI, bias, scale):
    """
    Double-logistic function for fitting NDVI profiles.
    """
    spring = 1 / (1 + np.exp(-np.sin((t-xmidSNDVI)*2*np.pi/365) / scalSNDVI))
    return bias + scale * (spring )


# In[144]:


# Define initial guesses
bias_guess = 1000  # Example bias
scale_guess = 5000  # Example scale
initial_guess = [320,1,4500, 20000 ]  # Northern hemisphere


# In[145]:


# Generate fitted curve
doy_full = np.arange(1, 366)  # Full year for smooth plotting
ndvi_fitted = double_sigmoid_sine(doy_full, *initial_guess)

# Plot fitted curve
plt.figure(figsize=(10, 5))
plt.scatter(doy_values, ndvi_values, color="black", label="Observed NDVI")
plt.plot(doy_full, ndvi_fitted, color="red", linestyle="--", label="Fitted Double-Logistic Curve")


plt.xlabel("Day of Year")
plt.ylabel("NDVI")
plt.title(f"NDVI Seasonal Curve Fitting" )
plt.legend()
plt.grid(True)
plt.show()


# In[146]:


# Fit the double-logistic function
params, _ = curve_fit(double_sigmoid_sine, doy_values, ndvi_values, p0=initial_guess, maxfev=20000)

# Generate fitted curve
doy_full = np.arange(-700, 700)  # Full year for smooth plotting
ndvi_fitted = double_sigmoid_sine(doy_full, *params)

# Plot fitted curve
plt.figure(figsize=(10, 5))
plt.scatter(doy_values, ndvi_values, color="black", label="Observed NDVI")
plt.plot(doy_full, ndvi_fitted, color="red", linestyle="--", label="Fitted Double-Logistic Curve")


plt.xlabel("Day of Year")
plt.ylabel("NDVI")
plt.title(f"NDVI Seasonal Curve Fitting" )
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




