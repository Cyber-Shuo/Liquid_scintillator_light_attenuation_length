import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib import cm
from scipy.optimize import curve_fit
import pandas as pd

AL_value, error_value, sample_name = [],[],[]
lines = np.loadtxt('AL_DATA_PATH', dtype = str, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)
for line in lines:
    AL_value.append(float(line[0]))
    error_value.append(float(line[1]))
    sample_name.append(line[2])
print(AL_value, error_value, sample_name)

fig, ax1 = plt.subplots()
ax1.set_xlabel('Attenuation length/m')
ax1.set_ylabel('Sample Name')

norm = plt.Normalize(20,35)
map_vir = cm.get_cmap(name = 'Blues')
colors = map_vir(norm(AL_value))

ax1.barh(sample_name, AL_value, color=colors, xerr = error_value, error_kw = {'ecolor':'0.2', 'capsize':3, 'capthick':1, 'mfc':'r'}, hatch = '', height = 0.3, label = 'Attenuation length')

sm = cm.ScalarMappable(cmap = map_vir, norm = norm)

for a,b,c in zip(AL_value, sample_name, error_value):  
    ax1.text(10, b, a, ha='center', va='center', fontsize=11, weight = 'semibold', color = 'black')

plt.legend(loc = "upper right")
# plt.colorbar(sm)
plt.show()
