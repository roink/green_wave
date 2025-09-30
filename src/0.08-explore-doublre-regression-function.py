#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# In[2]:


def double_logistic(t, xmidSNDVI, scalSNDVI, xmidANDVI, scalANDVI, bias, scale):
    """
    Double-logistic function for fitting NDVI profiles.
    """
    spring = 1 / (1 + np.exp((xmidSNDVI - t) / scalSNDVI))
    autumn = 1 / (1 + np.exp((xmidANDVI - t) / scalANDVI))
    return bias + scale * (spring - autumn)


# In[3]:


# Define the time range (DOY 0-365)
t_values = np.linspace(0, 365, 366)

# Define initial guesses
bias_guess = 1000  # Example bias
scale_guess = 5000  # Example scale
initial_guess_1 = [120, 20, 270, 25, bias_guess, scale_guess]  # Northern hemisphere
initial_guess_2 = [240, 20, 60, 25, bias_guess+scale_guess, scale_guess]   # Southern hemisphere
initial_guess_3 = [240, 20, 60, 25, bias_guess+0.5*scale_guess, 0]   # equatorial hemisphere


# Compute function values for both initial guesses
ndvi_north = double_logistic(t_values, *initial_guess_1)
ndvi_south = double_logistic(t_values, *initial_guess_2)
ndvi_equatorial = double_logistic(t_values, *initial_guess_3)



# Plot the functions
plt.figure(figsize=(10, 5))
plt.plot(t_values, ndvi_north, label="Northern Hemisphere", color="green")
plt.plot(t_values, ndvi_south, label="Southern Hemisphere", color="blue")
plt.plot(t_values, ndvi_equatorial, label="Equatorial", color="red")

plt.xlabel("Day of Year (DOY)")
plt.ylabel("NDVI")
plt.title("Double-Logistic NDVI Function for Different Initial Guesses")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




