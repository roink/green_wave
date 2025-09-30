#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# Function to warp time
def g(x, T, p):
    return x**p / (x**p + (T - x)**p)

# Generate x values
T = 2 * np.pi  # Full period
x = np.linspace(-T, 2*T, 1000)

# Different values of p
p_values = [0.5, 1, 2, 5]

# Plot the warped sine function
plt.figure(figsize=(8, 6))

for p in p_values:
    warped_x = g(x, T, p) * T  # Transform x using g(x)
    y = np.sin(2 * np.pi * warped_x / T)  # Compute sine
    plt.plot(x, y, label=f'p={p}')

plt.xlabel('x')
plt.ylabel('sin(warped x)')
plt.title('Warped Sine Function with Different p Values')
plt.legend()
plt.grid()
plt.show()


# In[20]:


import numpy as np
import matplotlib.pyplot as plt


# Generate x values
T = 1  # Full period
x = np.linspace(0, T, 1000)

# Different values of p
p_values = [0.5, 1]

# Plot the warped sine function
plt.figure(figsize=(8, 6))

for p in p_values:
    warped_x = g(x, T, p) * T  # Transform x using g(x)
    y = np.sin(2 * np.pi * x + p * np.sin(2*np.pi *(x-t2)))  # Compute sine
    plt.plot(x, y, label=f'p={p}')

plt.xlabel('x')
plt.ylabel('sin(warped x)')
plt.title('Warped Sine Function with Different p Values')
plt.legend()
plt.grid()
plt.show()


# In[23]:


import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

def warped_sine_plot(p=1.0, t2=0.0):
    # Generate x values
    T = 1  # Full period
    x = np.linspace(0, T, 1000)
    
    # Compute warped sine function
    y = np.sin(2 * np.pi * x + p * np.sin(2 * np.pi * (x - t2)))
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=f'p={p}, t2={t2}')
    plt.xlabel('x')
    plt.ylabel('sin(warped x)')
    plt.title('Warped Sine Function with Interactive Parameters')
    plt.legend()
    plt.grid()
    plt.show()

# Create interactive sliders
interact(warped_sine_plot, 
         p=FloatSlider(min=0., max=1.0, step=0.01, value=0.5, description='p'),
         t2=FloatSlider(min=0, max=1.0, step=0.01, value=0.0, description='t2'))


# In[31]:


import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

def sigmoid(x):
    return 1/(1+np.exp(x))

def warped_sine_plot(p=1.0, t2=0.0, a=1.0,b=0.0):
    # Generate x values
    T = 1  # Full period
    x = np.linspace(0, T, 1000)
    
    # Compute warped sine function
    y = sigmoid(a*np.sin(2 * np.pi * x + p * np.sin(2 * np.pi * (x - t2)))+b)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=f'p={p}, t2={t2}')
    plt.xlabel('x')
    plt.ylabel('sin(warped x)')
    plt.title('Warped Sine Function with Interactive Parameters')
    plt.legend()
    plt.grid()
    plt.show()

# Create interactive sliders
interact(warped_sine_plot, 
         p=FloatSlider(min=0., max=1.0, step=0.01, value=0.5, description='p'),
         t2=FloatSlider(min=0, max=1.0, step=0.01, value=0.0, description='t2'),
         a=FloatSlider(min=0, max=100.0, step=0.1, value=1.0, description='a'),
         b=FloatSlider(min=-100, max=100.0, step=0.1, value=0.0, description='b'))


# In[ ]:




