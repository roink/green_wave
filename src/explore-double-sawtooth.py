#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def sawtooth(t,period):
    return t % period


# In[3]:


period = 365

# Generate time points over multiple periods
t_values = np.linspace(0, 2 * period, 1000)
f_values = sawtooth(t_values, period)

# Plot the function
plt.figure(figsize=(10, 5))
plt.plot(t_values, f_values, label="Periodic Linear Function", color="b")
plt.axhline(0, color="k", linestyle="--", linewidth=1)
plt.xlabel("t (days)")
plt.ylabel("f(t)")
plt.title("Piecewise-Linear Periodic Function")
plt.legend()
plt.grid(True)
plt.show()


# In[4]:


def sawtooth_zp(t,period,zero_point):
    return ((t-zero_point-period/2) % period)-period/2


# In[5]:


# Generate time points over multiple periods
t_values = np.linspace(0, 2 * period, 1000)
f_values = sawtooth_zp(t_values, period,120)

# Plot the function
plt.figure(figsize=(10, 5))
plt.plot(t_values, f_values, label="Periodic Linear Function", color="b")
plt.axhline(0, color="k", linestyle="--", linewidth=1)
plt.xlabel("t (days)")
plt.ylabel("f(t)")
plt.title("Piecewise-Linear Periodic Function")
plt.legend()
plt.grid(True)
plt.show()


# In[6]:


def sawtooth_zp_d(t,period,zero_point,d):
    return (((t-zero_point-period/2) % period)-period/2)*d


# In[7]:


# Generate time points over multiple periods
t_values = np.linspace(0, 2 * period, 1000)
f_values = sawtooth_zp_d(t_values, period,120,2)

# Plot the function
plt.figure(figsize=(10, 5))
plt.plot(t_values, f_values, label="Periodic Linear Function", color="b")
plt.axhline(0, color="k", linestyle="--", linewidth=1)
plt.xlabel("t (days)")
plt.ylabel("f(t)")
plt.title("Piecewise-Linear Periodic Function")
plt.legend()
plt.grid(True)
plt.show()


# In[8]:


# Generate time points over multiple periods
t_values = np.linspace(0, 2 * period, 1000)
f_values = sawtooth_zp_d(t_values, period,120,2)
f2_values = sawtooth_zp_d(t_values, period,250,-3)

# Plot the function
plt.figure(figsize=(10, 5))
plt.plot(t_values, f_values, label="Increase", color="b")
plt.plot(t_values, f2_values, label="Decrease", color="r")

plt.axhline(0, color="k", linestyle="--", linewidth=1)
plt.xlabel("t (days)")
plt.ylabel("f(t)")
plt.title("Piecewise-Linear Periodic Function")
plt.legend()
plt.grid(True)
plt.show()


# In[9]:


# Given parameters
period = 365  # Example period
z1 = 120  # Zero point of first sawtooth
d1 = 2  # Slope scaling factor of first sawtooth

z2 = 250  # Zero point of second sawtooth
d2 = -3  # Slope scaling factor of second sawtooth

# Compute first few crossing points analytically

t_crossings = (z1 * d1 - z2 * d2 ) / (d1 - d2)
t_crossings


# In[10]:


# Compute first few crossing points analytically

t_crossings = ((z1+365) * d1 - z2 * d2 ) / (d1 - d2)
t_crossings


# In[11]:


# Compute first few crossing points analytically

t_crossings = ((z1-365) * d1 - z2 * d2 ) / (d1 - d2)
t_crossings


# In[25]:


import numpy as np

def double_sawtooth(t, period, z1, d1, z2, d2):
    st1 = (((t - z1 - period / 2) % period) - period / 2) * d1
    st2 = (((t - z2 - period / 2) % period) - period / 2) * d2
    t1 = (z1 * d1 - z2 * d2) / (d1 - d2)
    t2 = ((z1 + 365) * d1 - z2 * d2) / (d1 - d2)
    
    # Corrected condition using element-wise logical operations
    condition = ((t % period) > t1) & ((t % period) < t2)
    
    return np.where(condition, st2, st1)  #


# In[26]:


# Generate time points over multiple periods
t_values = np.linspace(0, 2 * period, 1000)

f_values = double_sawtooth(t_values, period,z1,d1,z2,d2)
t1 = (z1 * d1 - z2 * d2) / (d1 - d2)
t2 = ((z1 + 365) * d1 - z2 * d2) / (d1 - d2)
print(t1)
print(t2)

# Plot the function
plt.figure(figsize=(10, 5))
plt.plot(t_values, f_values, label="Increase", color="b")

plt.axhline(0, color="k", linestyle="--", linewidth=1)
plt.xlabel("t (days)")
plt.ylabel("f(t)")
plt.title("Piecewise-Linear Periodic Function")
plt.legend()
plt.grid(True)
plt.show()


# In[27]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[32]:


period = 365  # Example period
z1 = 120  # Zero point of first sawtooth
d1 = 1/20  # Slope scaling factor of first sawtooth

z2 = 250  # Zero point of second sawtooth
d2 = 1/-30  # Slope scaling factor of second sawtooth

# Generate time points over multiple periods
t_values = np.linspace(0, 2 * period, 1000)

f_values = sigmoid(double_sawtooth(t_values, period,z1,d1,z2,d2))
f_values2 = double_sawtooth(t_values, period,z1,d1,z2,d2)

t1 = (z1 * d1 - z2 * d2) / (d1 - d2)
t2 = ((z1 + 365) * d1 - z2 * d2) / (d1 - d2)
print(t1)
print(t2)

# Plot the function
plt.figure(figsize=(10, 5))
plt.plot(t_values, f_values, label="Increase", color="b")
plt.plot(t_values, f_values2, label="Increase", color="r")

plt.axhline(0, color="k", linestyle="--", linewidth=1)
plt.xlabel("t (days)")
plt.ylabel("f(t)")
plt.title("Piecewise-Linear Periodic Function")
plt.legend()
plt.grid(True)
plt.show()


# In[47]:


z1 = 250  # Zero point of first sawtooth
d1 = 1/20  # Slope scaling factor of first sawtooth

z2 = 120  # Zero point of second sawtooth
d2 = 1/-30  # Slope scaling factor of second sawtooth

def double_sawtooth(t, period, z1, d1, z2, d2):
    st1 = (((t - z1 - period / 2) % period) - period / 2) * d1
    st2 = (((t - z2 - period / 2) % period) - period / 2) * d2
    t1 = (z1 * d1 - z2 * d2) / (d1 - d2)
    t2 = ((z1 + 365) * d1 - z2 * d2) / (d1 - d2)
    print(t1)
    print(t2)
    # Corrected condition using element-wise logical operations
    if t2<t1:
        if z2> z1:
            condition = ((t % period) > t2) & ((t % period) < (t1 ))
    
            return np.where(condition, st1, st2)  #
        else: 
            condition = ((t % period) > t2) & ((t % period) < (t1 ))
            return np.where(condition, st1, st2)  #
    else: 
        condition = ((t % period) > t1) & ((t % period) < (t2 ))
        return np.where(condition, st2, st1)

# Generate time points over multiple periods
t_values = np.linspace(0, 2 * period, 1000)
f_values = sawtooth_zp_d(t_values, period,z1,d1)
f2_values = sawtooth_zp_d(t_values, period,z2,d2)

f_values2 = double_sawtooth(t_values, period,z1,d1,z2,d2)




# Plot the function
plt.figure(figsize=(10, 5))
plt.plot(t_values, f_values, label="Increase", color="b")
plt.plot(t_values, f2_values, label="Decrease", color="r")

plt.plot(t_values, f_values2, label="Decrease", color="g")

plt.axhline(0, color="k", linestyle="--", linewidth=1)
plt.xlabel("t (days)")
plt.ylabel("f(t)")
plt.title("Piecewise-Linear Periodic Function")
plt.legend()
plt.grid(True)
plt.show()


# In[60]:


def find_sawtooth_intersections(z1, d1, z2, d2, period=365):
    """
    Find the two intersection points t in [0, period) where
        f1(t) = sawtooth_zp_d(t, period, z1, d1)
        f2(t) = sawtooth_zp_d(t, period, z2, d2)
    using the analytic approach.
    
    Parameters
    ----------
    z1, z2 : float
        The horizontal "zero crossing" offsets for the two sawtooth functions.
    d1, d2 : float
        The slope factors of the two sawtooth functions (d1 > 0, d2 < 0).
    period : float, optional
        The period of the sawtooth (default 365).
    
    Returns
    -------
    list of float
        Up to two intersection points t in the interval [0, period), sorted ascending.
    """
    # Wrap z1, z2 into [0, period)
    z1_mod = z1 % period
    z2_mod = z2 % period
    
    # r = d2/d1
    r = d2 / d1  # negative, since d2 < 0 and d1 > 0
    
    # d' = (z1 - z2) mod P, in [0, period)
    dprime = (z1_mod - z2_mod) % period
    
    # -- Case 1: T1
    #   T1 = [r*d' + (P/2)*(1 - r)] / (1 - r)
    #   valid if 0 <= T1 < (P - d')
    numerator_1 = r*dprime + (period/2)*(1 - r)
    denominator = (1 - r)
    T1 = numerator_1 / denominator  # candidate
    
    # -- Case 2: T2
    #   T2 = [r*d' + (P/2)*(1 - 3r)] / (1 - r)
    #   valid if (P - d') <= T2 < P
    numerator_2 = r*dprime + (period/2)*(1 - 3*r)
    T2 = numerator_2 / denominator  # candidate
    
    # Collect valid solutions
    solutions = []
    
    # Check if T1 is in [0, P - dprime)
    print("T1:", T1)
    if 0 <= T1 < (period - dprime):
        t1 = (z1_mod + (period/2) + T1) % period
        solutions.append(t1)
        print("Appended solution t1:", t1)
    
    # Check if T2 is in [P - dprime, P)
    print("T2:", T2)
    if (period - dprime) <= T2 < period:
        t2 = (z1_mod + (period/2) + T2) % period
        solutions.append(t2)
        print("Appended solution t2:", t2)
    
    # Sort to return them in ascending order
    print("Solutions before sorting", solutions)
    solutions.sort()
    print("Solutions after sorting", solutions)
    return solutions


    
intersections = find_sawtooth_intersections(z1, d1, z2, d2, period)
print("Intersections within [0, 365):", intersections)


# In[53]:


z1 = 250  # Zero point of first sawtooth
d1 = 1/20  # Slope scaling factor of first sawtooth

z2 = 120  # Zero point of second sawtooth
d2 = 1/-10  # Slope scaling factor of second sawtooth


# Generate time points over multiple periods
t_values = np.linspace(0, 2 * period, 1000)
f_values = sawtooth_zp_d(t_values, period,z1,d1)
f2_values = sawtooth_zp_d(t_values, period,z2,d2)

solutions = find_sawtooth_intersections(z1, d1, z2, d2, period=365)

# Plot the function
plt.figure(figsize=(10, 5))
plt.plot(t_values, f_values, label="Increase", color="b")
plt.plot(t_values, f2_values, label="Decrease", color="r")


plt.axhline(0, color="k", linestyle="--", linewidth=1)
plt.axvline(solutions[0], color="green", linestyle=":", label="Spring Inflection")
plt.axvline(solutions[1], color="orange", linestyle=":", label="Autumn Inflection")
plt.xlabel("t (days)")
plt.ylabel("f(t)")
plt.title("Piecewise-Linear Periodic Function")
plt.legend()
plt.grid(True)
plt.show()


# In[61]:


def double_sawtooth(t, period, z1, d1, z2, d2):
    """
    Piecewise function that:
      - uses sawtooth #1 (z1,d1) outside the two crossing times,
      - uses sawtooth #2 (z2,d2) between them.
    This repeats every `period`.
    
    Parameters
    ----------
    t : array-like
        Time values at which to evaluate the piecewise function.
    period : float
        The period of the sawtooth, e.g. 365.
    z1, d1 : float
        The zero-cross offset and slope of the first sawtooth (positive slope).
    z2, d2 : float
        The zero-cross offset and slope of the second sawtooth (negative slope).
    
    Returns
    -------
    numpy array
        The values of the piecewise "double sawtooth" function at times t.
    """

    t = np.asarray(t)  # ensure array for vectorized operations

    # -- 1) Evaluate each sawtooth individually
    # sawtooth_zp_d(t,period,z) = [((t - z - period/2) % period) - period/2] * d
    st1 = (((t - z1 - period / 2) % period) - period / 2) * d1
    st2 = (((t - z2 - period / 2) % period) - period / 2) * d2
    
    # -- 2) Find the crossing times
    crossing_times = find_sawtooth_intersections(z1, d1, z2, d2, period)
    print(crossing_times)
    
    # If for some reason we get fewer than 2 intersections (unusual with d1>0, d2<0),
    # handle gracefully by returning one of them, say st1.
    if len(crossing_times) < 2:
        return st1
    
    c1, c2 = crossing_times[0], crossing_times[1]
    
    # -- 3) Define a boolean mask for the middle region in each period,
    #       i.e. when t%period is in [c1, c2).
    #       We'll select st2 in that region, and st1 otherwise.
    modt = t % period  # we only care about fractional part within 1 period
    condition_middle = (modt >= c1) & (modt < c2)
    
    # st2 in the middle region, st1 elsewhere
    out = np.where(condition_middle, st2, st1)
    return out

# --------------- EXAMPLE USAGE ---------------

if True:

    period = 365
    z1, d1 = 250, 2
    z2, d2 = 120, -1

    # A range of time to plot (e.g., a bit more than one period)
    t_vals = np.linspace(0, 2 * period, 1000)

    # Evaluate the "double sawtooth"
    ds_vals = double_sawtooth(t_vals, period, z1, d1, z2, d2)

    # Also for reference, evaluate each sawtooth alone
    st1_vals = (((t_vals - z1 - period/2) % period) - period/2) * d1
    st2_vals = (((t_vals - z2 - period/2) % period) - period/2) * d2

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(t_vals, st1_vals, label="Sawtooth 1 (z1,d1)", alpha=0.5)
    plt.plot(t_vals, st2_vals, label="Sawtooth 2 (z2,d2)", alpha=0.5)
    plt.plot(t_vals, ds_vals, 'k-', label="Double Sawtooth", linewidth=2)
    plt.title("Double Sawtooth vs. Individual Sawtooths")
    plt.legend()
    plt.grid(True)
    plt.show()


# In[62]:


def sawtooth_zp_d(t, period, z, d):
    """
    The basic sawtooth function used:
       f(t) = [((t - z - period/2) % period) - period/2] * d
    """
    return (((t - z - period/2) % period) - period/2) * d

def find_sawtooth_intersections(z1, d1, z2, d2, period=365):
    """
    Analytically find up to two intersection points t in [0, period)
    where sawtooth_zp_d(t,period,z1,d1) == sawtooth_zp_d(t,period,z2,d2).
    Returns a sorted list (usually 2 values) of crossing times.
    """
    z1_mod = z1 % period
    z2_mod = z2 % period
    r = d2 / d1  # typically < 0
    dprime = (z1_mod - z2_mod) % period

    # Candidate T1
    num1 = r*dprime + (period/2)*(1 - r)
    den = (1 - r)
    T1 = num1 / den  # candidate

    # Candidate T2
    num2 = r*dprime + (period/2)*(1 - 3*r)
    T2 = num2 / den  # candidate

    solutions = []
    # Case 1: T1 valid if 0 <= T1 < (P - d')
    if 0 <= T1 < (period - dprime):
        c1 = (z1_mod + (period/2) + T1) % period
        solutions.append(c1)
    # Case 2: T2 valid if (P - d') <= T2 < P
    if (period - dprime) <= T2 < period:
        c2 = (z1_mod + (period/2) + T2) % period
        solutions.append(c2)

    solutions.sort()
    return solutions

def double_sawtooth(t, period, z1, d1, z2, d2):
    """
    Piecewise function:
      - We find the two crossing times c1 < c2 in [0, period).
      - We check which function is smaller at t%period=0.
      - The function uses that 'initial function' from 0..c1,
        then switches from c1..c2,
        then switches back from c2..period,
        repeating every period.
    """
    t = np.asarray(t)  # vectorize
    st1 = sawtooth_zp_d(t, period, z1, d1)
    st2 = sawtooth_zp_d(t, period, z2, d2)

    # Get crossing times
    crossings = find_sawtooth_intersections(z1, d1, z2, d2, period)
    if len(crossings) < 2:
        # fallback: if for some reason there's < 2, just return st1
        return st1
    c1, c2 = crossings

    # Evaluate st1(0), st2(0) to see which is smaller at t=0 mod P
    val1_0 = sawtooth_zp_d(0, period, z1, d1)
    val2_0 = sawtooth_zp_d(0, period, z2, d2)
    st1_first = (val1_0 < val2_0)  # True if st1 is smaller at t=0

    modt = t % period

    # We'll build the output in 3 intervals: [0,c1), [c1,c2), [c2,P)
    # If st1_first is True, then st1 in the first/third intervals, st2 in the second
    # If st1_first is False, then st2 in the first/third intervals, st1 in the second

    if st1_first:
        # st1 in first, st2 in second, st1 in third
        cond_first  = (modt < c1)
        cond_second = (modt >= c1) & (modt < c2)
        # else third
        out = np.where(cond_first, st1,
               np.where(cond_second, st2, st1))
    else:
        # st2 in first, st1 in second, st2 in third
        cond_first  = (modt < c1)
        cond_second = (modt >= c1) & (modt < c2)
        out = np.where(cond_first, st2,
               np.where(cond_second, st1, st2))

    return out


# In[65]:


if True:

    period = 365
    z1, d1 = 120, 2
    z2, d2 = 2500, -3

    # A range of time to plot (e.g., a bit more than one period)
    t_vals = np.linspace(0, 2 * period, 1000)

    # Evaluate the "double sawtooth"
    ds_vals = double_sawtooth(t_vals, period, z1, d1, z2, d2)

    # Also for reference, evaluate each sawtooth alone
    st1_vals = (((t_vals - z1 - period/2) % period) - period/2) * d1
    st2_vals = (((t_vals - z2 - period/2) % period) - period/2) * d2

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(t_vals, st1_vals, label="Sawtooth 1 (z1,d1)", alpha=0.5)
    plt.plot(t_vals, st2_vals, label="Sawtooth 2 (z2,d2)", alpha=0.5)
    plt.plot(t_vals, ds_vals, 'k-', label="Double Sawtooth", linewidth=2)
    plt.title("Double Sawtooth vs. Individual Sawtooths")
    plt.legend()
    plt.grid(True)
    plt.show()


# In[69]:


import numpy as np
import matplotlib.pyplot as plt

def sawtooth_zp_d(t, period, z, d):
    """
    Basic sawtooth:
      saw(t) = [((t - z - period/2) % period) - period/2]* d
    """
    return (((t - z - period/2) % period) - period/2) * d

def find_sawtooth_intersections(z1, d1, z2, d2, period=365):
    """
    Analytically find up to two intersection points t in [0, period)
    where sawtooth_zp_d(t,period,z1,d1) == sawtooth_zp_d(t,period,z2,d2).

    Returns a sorted list (usually exactly 2 values) c1 < c2.
    """
    z1_mod = z1 % period
    z2_mod = z2 % period

    # r = d2 / d1
    r = d2 / d1  # typically negative if d1>0,d2<0
    dprime = (z1_mod - z2_mod) % period

    # Candidate T1
    num1 = r*dprime + (period/2)*(1 - r)
    den  = (1 - r)
    T1 = num1 / den

    # Candidate T2
    num2 = r*dprime + (period/2)*(1 - 3*r)
    T2 = num2 / den

    solutions = []
    # Case 1 valid: T1 in [0, P-dprime)
    if 0 <= T1 < (period - dprime):
        c1 = (z1_mod + (period/2) + T1) % period
        solutions.append(c1)
    # Case 2 valid: T2 in [P-dprime, P)
    if (period - dprime) <= T2 < period:
        c2 = (z1_mod + (period/2) + T2) % period
        solutions.append(c2)

    solutions.sort()  # c1 < c2 if both exist
    return solutions

def double_sawtooth(t, period, z1, d1, z2, d2):
    """
    Construct the piecewise function with 2 crossing points c1<c2 in [0,period):
      Let val1 = st1(c1) = st2(c1),
          val2 = st1(c2) = st2(c2).
      
      - If val1 < val2:
          c1 is the 'lowest crossing', c2 is the 'highest crossing'.
          => st2 on [0, c1) and [c2, P),
             st1 on [c1, c2).
             
      - If val1 > val2:
          c1 is the 'highest crossing', c2 is the 'lowest crossing'.
          => st1 on [0, c1) and [c2, P),
             st2 on [c1, c2).
    """
    # Convert t to numpy array for vectorized operations
    t = np.asarray(t)

    # Evaluate each sawtooth
    st1 = sawtooth_zp_d(t, period, z1, d1)  # increasing
    st2 = sawtooth_zp_d(t, period, z2, d2)  # decreasing

    # Get the crossing times
    crossings = find_sawtooth_intersections(z1, d1, z2, d2, period)
    if len(crossings) < 2:
        # If for some reason fewer than 2 are found, just return st1
        return st1
    
    c1, c2 = crossings[0], crossings[1]
    
    # Evaluate the crossing values (same for st1 or st2 at those times)
    val1 = sawtooth_zp_d(c1, period, z1, d1)  # st1(c1) = st2(c1)
    val2 = sawtooth_zp_d(c2, period, z1, d1)  # st1(c2) = st2(c2)

    # We'll define piecewise intervals in ascending time: [0,c1), [c1,c2), [c2,P)
    # then repeat periodically. So let's get the fractional part t % period:
    modt = t % period

    # Build output in three intervals using np.where
    # We'll define a boolean array for each interval:
    in_first  = (modt >= 0)   & (modt < c1)
    in_second = (modt >= c1)  & (modt < c2)
    # The rest is the third interval: (modt >= c2) or modt < 0 (but that won't happen)
    
    out = np.zeros_like(t)

    if val1 < val2:
        # CASE A: c1 is the 'lowest crossing', c2 the 'highest crossing'
        # => st2 on [0, c1), st1 on [c1, c2), st2 on [c2, P)
        out[in_first]  = st2[in_first]
        out[in_second] = st1[in_second]
        # outside second => st2
        out[~(in_first | in_second)] = st2[~(in_first | in_second)]
    else:
        # CASE B: c1 is the 'highest crossing', c2 the 'lowest crossing'
        # => st1 on [0, c1), st2 on [c1, c2), st1 on [c2, P)
        out[in_first]  = st1[in_first]
        out[in_second] = st2[in_second]
        # outside second => st1
        out[~(in_first | in_second)] = st1[~(in_first | in_second)]

    return out


# --------------- DEMO ---------------
if __name__ == "__main__":
    period = 365
    # Example #1: typical
    z1, d1 = 120,  2   # increasing sawtooth
    z2, d2 = 250, -1   # decreasing sawtooth

    # Evaluate from t=0 up to 2*period to see repeating shape
    t_vals = np.linspace(0, 2*period, 1200)
    ds_vals = double_sawtooth(t_vals, period, z1, d1, z2, d2)

    # For reference, each sawtooth alone:
    st1_vals = sawtooth_zp_d(t_vals, period, z1, d1)
    st2_vals = sawtooth_zp_d(t_vals, period, z2, d2)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,6))
    plt.plot(t_vals, st1_vals, label="st1 (increasing)", alpha=0.5)
    plt.plot(t_vals, st2_vals, label="st2 (decreasing)", alpha=0.5)
    plt.plot(t_vals, ds_vals, 'k-', label="Double Sawtooth", linewidth=2)
    plt.title("Double Sawtooth with 'lowest-to-highest = st1' rule")
    plt.legend()
    plt.grid(True)
    plt.show()


# In[82]:


period = 365  # Example period
z1 = 280  # Zero point of first sawtooth
d1 = 1/10  # Slope scaling factor of first sawtooth

z2 = 10  # Zero point of second sawtooth
d2 = 1/-10  # Slope scaling factor of second sawtooth

# Generate time points over multiple periods
t_values = np.linspace(0, 2 * period, 1000)

f_values = sigmoid(double_sawtooth(t_values, period,z1,d1,z2,d2))
f_values2 = f_values * (1-f_values)

t1 = (z1 * d1 - z2 * d2) / (d1 - d2)
t2 = ((z1 + 365) * d1 - z2 * d2) / (d1 - d2)
print(t1)
print(t2)

# Plot the function
plt.figure(figsize=(10, 5))
plt.plot(t_values, f_values, label="NDVI", color="b")
plt.plot(t_values, f_values2, label="IRG", color="r")

plt.axhline(0, color="k", linestyle="--", linewidth=1)
plt.xlabel("t (days)")
plt.ylabel("f(t)")
plt.title("Piecewise-Linear Periodic Function")
plt.legend()
plt.grid(True)
plt.show()


# In[94]:


import numpy as np
import matplotlib.pyplot as plt

def sawtooth_zp_d(t, period, z, d):
    """
    The basic sawtooth:
      f(t) = [((t - z - period/2) % period) - period/2] * d
    """
    return (((t - z - period/2) % period) - period/2) * d

def find_sawtooth_intersections(z1, d1, z2, d2, period=365):
    """
    Analytically find up to two intersection points t in [0, period)
    where sawtooth_zp_d(t,period,z1,d1) == sawtooth_zp_d(t,period,z2,d2).

    Returns a sorted list (usually exactly 2 values) c1 < c2.
    """
    z1_mod = z1 % period
    z2_mod = z2 % period

    # r = d2 / d1
    r = d2 / d1  # typically negative if d1>0, d2<0
    dprime = (z1_mod - z2_mod) % period

    # Candidate T1
    num1 = r*dprime + (period/2)*(1 - r)
    den  = (1 - r)
    T1 = num1 / den

    # Candidate T2
    num2 = r*dprime + (period/2)*(1 - 3*r)
    T2 = num2 / den

    solutions = []
    # Case 1 valid if 0 <= T1 < (P - d')
    if 0 <= T1 < (period - dprime):
        c1 = (z1_mod + (period/2) + T1) % period
        solutions.append(c1)
    # Case 2 valid if (P - dprime) <= T2 < period
    if (period - dprime) <= T2 < period:
        c2 = (z1_mod + (period/2) + T2) % period
        solutions.append(c2)

    solutions.sort()
    return solutions

def double_sawtooth(t, period, z1, d1, z2, d2):
    """
    Piecewise combination of two sawtooths st1 and st2:
      - st1 is slope d1 (increasing)
      - st2 is slope d2 (decreasing)
    with two crossings in [0, period). 
    This function returns the piecewise value x(t).
    """
    t = np.asarray(t)
    st1 = sawtooth_zp_d(t, period, z1, d1)  # increasing
    st2 = sawtooth_zp_d(t, period, z2, d2)  # decreasing

    crossings = find_sawtooth_intersections(z1, d1, z2, d2, period)
    if len(crossings) < 2:
        # fallback if fewer than 2 intersections
        return st1

    c1, c2 = crossings

    # Evaluate the crossing function values (the same for st1 or st2 at each crossing)
    val1 = sawtooth_zp_d(c1, period, z1, d1)
    val2 = sawtooth_zp_d(c2, period, z1, d1)

    modt = (t % period)
    in_first  = (modt < c1)
    in_second = (modt >= c1) & (modt < c2)

    out = np.zeros_like(t)

    # If val1 < val2 => c1 is the "lowest crossing", so st1 is used between c1..c2
    # otherwise st1 is used outside that interval.
    if val1 < val2:
        # st2 for [0,c1) and [c2,P), st1 for [c1,c2)
        out[in_first]  = st2[in_first]
        out[in_second] = st1[in_second]
        out[~(in_first | in_second)] = st2[~(in_first | in_second)]
    else:
        # st1 for [0,c1) and [c2,P), st2 for [c1,c2)
        out[in_first]  = st1[in_first]
        out[in_second] = st2[in_second]
        out[~(in_first | in_second)] = st1[~(in_first | in_second)]

    return out

def double_sawtooth_slope(t, period, z1, d1, z2, d2):
    """
    Return x'(t), i.e. the piecewise slope of double_sawtooth(t).
    - It's simply d1 or d2 depending on which segment is active.
    """
    t = np.asarray(t)
    modt = (t % period)

    # Find the same crossing times
    crossings = find_sawtooth_intersections(z1, d1, z2, d2, period)
    if len(crossings) < 2:
        # fallback: slope is always d1
        return np.full_like(t, d1)

    c1, c2 = crossings

    val1 = sawtooth_zp_d(c1, period, z1, d1)
    val2 = sawtooth_zp_d(c2, period, z1, d1)

    in_first  = (modt < c1)
    in_second = (modt >= c1) & (modt < c2)

    out = np.zeros_like(t)

    if val1 < val2:
        # same condition: st2 outside, st1 inside [c1,c2)
        out[in_first]  = d2
        out[in_second] = d1
        out[~(in_first | in_second)] = d2
    else:
        # st1 outside, st2 inside [c1,c2)
        out[in_first]  = d1
        out[in_second] = d2
        out[~(in_first | in_second)] = d1

    return out

def logistic(x):
    """Sigmoid function σ(x) = 1 / (1 + e^-x)."""
    return 1 / (1 + np.exp(-x))

def logistic_derivative(x):
    """Derivative of the sigmoid: σ'(x) = σ(x)*(1 - σ(x))."""
    s = logistic(x)
    return s * (1 - s)


# -------------- DEMO: PLOT SIGMOID(double_sawtooth) and ITS DERIVATIVE -------------
if True:
    # Example parameters
    period = 1
    z1, d1 = 250/365,  50/2   # st1: increasing
    z2, d2 = 120/365, 50/-1   # st2: decreasing

    # Time range to plot
    t_vals = np.linspace(0, 2*period, 1000)
    
    # 1) Evaluate x(t) = double_sawtooth(t)
    x_vals = double_sawtooth(t_vals, period, z1, d1, z2, d2)
    
    # 2) Evaluate σ(x(t))
    sig_vals = logistic(x_vals)
    
    # 3) Evaluate x'(t) (the piecewise slope)
    xprime_vals = double_sawtooth_slope(t_vals, period, z1, d1, z2, d2)
    
    # 4) Evaluate [d/dt] σ(x(t)) = σ'(x(t)) * x'(t)
    dsig_dt = logistic_derivative(x_vals) * xprime_vals
    
    # Plot:
    plt.figure(figsize=(10, 6))
    
    # Plot the function
    plt.plot(t_vals, sig_vals, label="sigmoid(double_sawtooth(t))", color="blue")
    
    # Plot its derivative
    plt.plot(t_vals, dsig_dt, label="d/dt [sigmoid(double_sawtooth(t))]", color="red")
    
    plt.title("Sigmoid(Double Sawtooth) and its Derivative")
    plt.xlabel("t")
    plt.grid(True)
    plt.legend()
    plt.show()


# In[104]:


import numpy as np

def sigmoid_double_sawtooth(
    t, z1, d1, z2, d2, bias, scale, period=365,return_derivative=False
):
    """
    Compute s(t) = sigmoid(double_sawtooth(t)) in one pass.
    Optionally also return ds/dt = d/dt [sigmoid(double_sawtooth(t))].

    Parameters
    ----------
    t : array-like
        Time points at which to evaluate.
    period : float
        The period (e.g. 365).
    z1, d1 : float
        Offset & slope for the increasing sawtooth (st1).
    z2, d2 : float
        Offset & slope for the decreasing sawtooth (st2).
    return_derivative : bool, optional
        If True, also return ds/dt in addition to s(t).

    Returns
    -------
    s_values : ndarray
        The values of sigmoid(double_sawtooth(t)).
    ds_dt_values : ndarray (only if return_derivative=True)
        The derivative d/dt [sigmoid(double_sawtooth(t))].
    """
    z1 = z1 % 365
    z2 = z2 % 365
    d1 = np.abs(d1)
    d2 = -np.abs(d2)
    t = np.asarray(t)

    # 1) Find the two crossing times in [0, period)
    c1, c2 = _get_crossings(z1, d1, z2, d2, period)

    # 2) Evaluate st1, st2 at the crossing times c1, c2 to see which is "lowest" or "highest"
    val1 = _sawtooth_zp_d(c1, period, z1, d1)
    val2 = _sawtooth_zp_d(c2, period, z1, d1)

    # We'll define the piecewise intervals:
    modt = t % period
    in_first  = (modt < c1)
    in_second = (modt >= c1) & (modt < c2)
    # The complement is the third interval: modt >= c2

    # Pre-allocate outputs
    x_vals = np.zeros_like(t)  # double_sawtooth(t)
    if return_derivative:
        xprime_vals = np.zeros_like(t)  # piecewise slope d1 or d2

    # 3) Fill in piecewise segments
    # If val1 < val2 => c1 is 'lowest crossing', c2 is 'highest crossing',
    # => st1 used on [c1,c2), st2 on [0,c1) & [c2,P).
    # Otherwise st2 used on [c1,c2), st1 on outside.
    if val1 < val2:
        # st2 in [0,c1) and [c2,P), st1 in [c1,c2)
        x_vals[in_first]  = _sawtooth_zp_d(t[in_first],  period, z2, d2)
        x_vals[in_second] = _sawtooth_zp_d(t[in_second], period, z1, d1)
        x_vals[~(in_first | in_second)] = _sawtooth_zp_d(t[~(in_first | in_second)], period, z2, d2)

        if return_derivative:
            xprime_vals[in_first]  = d2
            xprime_vals[in_second] = d1
            xprime_vals[~(in_first | in_second)] = d2
    else:
        # st1 in [0,c1) and [c2,P), st2 in [c1,c2)
        x_vals[in_first]  = _sawtooth_zp_d(t[in_first],  period, z1, d1)
        x_vals[in_second] = _sawtooth_zp_d(t[in_second], period, z2, d2)
        x_vals[~(in_first | in_second)] = _sawtooth_zp_d(t[~(in_first | in_second)], period, z1, d1)

        if return_derivative:
            xprime_vals[in_first]  = d1
            xprime_vals[in_second] = d2
            xprime_vals[~(in_first | in_second)] = d1

    # 4) Sigmoid
    s_vals = _logistic(x_vals)

    if return_derivative:
        # dsigma/dx = s * (1 - s)
        dsigma_dx = s_vals * (1 - s_vals)
        # By chain rule: d/dt [sigma(x(t))] = dsigma/dx * dx/dt
        ds_dt = dsigma_dx * xprime_vals
        return s_vals*scale+bias, ds_dt*scale

    else:
        # Just return the sigmoid
        return s_vals*scale+bias


# --- Small helpers (underscored to show they are "private") ---
def _sawtooth_zp_d(tt, period, z, d):
    """Compute the standard sawtooth at times tt, with offset z, slope d."""
    return (((tt - z - period/2) % period) - period/2) * d

def _logistic(x):
    """Sigmoid."""
    return 1 / (1 + np.exp(-x))

def _get_crossings(z1, d1, z2, d2, period=365):
    """
    Return the two crossing times c1, c2 in ascending order.
    If fewer than 2, just pick c1=0, c2=period/2 to avoid errors.
    """
    crossings = _find_sawtooth_intersections(z1, d1, z2, d2, period)
    if len(crossings) < 2:
        return (0.0, period / 2)
    return crossings[0], crossings[1]

def _find_sawtooth_intersections(z1, d1, z2, d2, period=365):
    """Analytic intersection solver, same logic as before."""
    z1_mod = z1 % period
    z2_mod = z2 % period

    r = d2 / d1
    dprime = (z1_mod - z2_mod) % period

    # T1
    num1 = r*dprime + (period/2)*(1 - r)
    den = (1 - r)
    T1 = num1 / den

    # T2
    num2 = r*dprime + (period/2)*(1 - 3*r)
    T2 = num2 / den

    solutions = []
    if 0 <= T1 < (period - dprime):
        c1 = (z1_mod + (period/2) + T1) % period
        solutions.append(c1)
    if (period - dprime) <= T2 < period:
        c2 = (z1_mod + (period/2) + T2) % period
        solutions.append(c2)

    solutions.sort()
    return solutions

period = 365  # Example period
z1 = 120  # Zero point of first sawtooth
d1 = 1/10  # Slope scaling factor of first sawtooth

z2 = 300  # Zero point of second sawtooth
d2 = 1/-10  # Slope scaling factor of second sawtooth

# Generate time points over multiple periods
t_values = np.linspace(0, 2 * period, 1000)

f_values = sigmoid_double_sawtooth(t_values,z1,d1,z2,d2,2500,6000)
f_values2 = f_values * (1-f_values)

t1 = (z1 * d1 - z2 * d2) / (d1 - d2)
t2 = ((z1 + 365) * d1 - z2 * d2) / (d1 - d2)
print(t1)
print(t2)

# Plot the function
plt.figure(figsize=(10, 5))
plt.plot(t_values, f_values, label="NDVI", color="b")
#plt.plot(t_values, f_values2, label="IRG", color="r")

plt.axhline(0, color="k", linestyle="--", linewidth=1)
plt.xlabel("t (days)")
plt.ylabel("f(t)")
plt.title("Piecewise-Linear Periodic Function")
plt.legend()
plt.grid(True)
plt.show()


# In[105]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# In[122]:


# True parameters
z1_true, d1_true = 250.0, .02
z2_true, d2_true = 120.0, -.01
bias_true = 1000
scale_true = 5000
period = 365

# Generate synthetic data
np.random.seed(123)
t_data = np.linspace(0, 2*period, 100)  # 100 points up to 2*period
y_clean = sigmoid_double_sawtooth(t_data, z1_true, d1_true, z2_true, d2_true, bias_true, scale_true, period)
noise = 0.05 * np.random.randn(len(t_data))  # 5% random noise
y_data = y_clean + noise


# In[123]:


def model_for_curve_fit(t, z1, d1, z2, d2, bias, scale):
    # curve_fit requires the model signature: f(t, *params).
    # We'll fix `period=365` inside or make it a global or partial function.
    return sigmoid_double_sawtooth(t, z1, d1, z2, d2,bias, scale,period=365)


# In[124]:


# Initial guess (z1, d1, z2, d2)
# Must guess sign for d2 to be negative if that's expected, etc.
p0 = [200, 0.1,  100, -0.05,0,1000]

# Optionally set parameter bounds:
#  e.g. we know z1,z2 in [0,365], d1>0, d2<0 (roughly).
# We'll put wide bounds but ensure d1>0, d2<0:

popt, pcov = curve_fit(
    model_for_curve_fit, 
    t_data, 
    y_data, 
    p0=p0,
)

print("Optimal parameters:", popt)
print("Estimated covariance matrix:\n", pcov)


# In[129]:


# Evaluate the fitted model
z1_fit, d1_fit, z2_fit, d2_fit,bias_fit, scale_fit = [ 9.33183349e+01 , 4.81735859e-02 , 3.55977364e+02 ,-2.11592117e-02,
  1.17265825e+03 , 7.14999646e+03]
y_fit = model_for_curve_fit(t_data, z1_fit, d1_fit, z2_fit, d2_fit, bias_fit, scale_fit)

plt.figure(figsize=(8,5))
plt.plot(t_data, y_fit, 'r-', label="Fitted Curve", linewidth=2)
plt.title("Fitting sigmoid(double_sawtooth(t)) with curve_fit")
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




