import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import pandas as pd

def ExpFunc(x, AL, ADC_0):
    return ADC_0*np.exp(-x/AL)

# get pedestal founction
def AL_getpedestal(path_initial_data, pedestal):
    # load gaussian fit data
    Alldata = np.loadtxt(path_initial_data)
    # get pedestal of last row
    pedestal = Alldata[-1,1]
    return pedestal

# get meandata founction
def AL_getmeandata(path_initial_data, path_mean_data):
    # load gaussian fit data
    Alldata = np.loadtxt(path_initial_data)
    # remove pedestal
    Alldata_remove_pedestal = Alldata[0:-2]
    # change to CSV data
    Excel_Alldata_remove_pedestal = pd.DataFrame(Alldata_remove_pedestal)
    # grouped by level data and averaged
    Excel_Alldata_remove_pedestal_mean = Excel_Alldata_remove_pedestal.groupby(0).mean()
    # change CSV data to txt file
    meandata = Excel_Alldata_remove_pedestal_mean.to_csv(path_mean_data, sep='\t', header = False)
    return path_mean_data

# data readout
def AL_writemeandata(path_mean_data, x_L_L, y_ADC, mean_sigma, y_erroy_bar):
    # open meandata and readout
    with open(path_mean_data,'r') as f:
        lines = f.readlines()
    # write level,ADC,error data to three list
    for line in lines:
        value = [float(s) for s in line.split()]
        x_L_L.append(value[0])
        y_ADC.append(value[1])
        mean_sigma.append(value[2])
    # get error bar data
    for i in y_ADC:
        y_error_bar.append(float((i-i*(0.9974))/y_ADC[0]))
    return 

def data_arrange(path_initial_data, pedestal, x_L_L, y_ADC, mean_sigma, norm_y_ADC_remove_pedestal, norm_fit_ADC_remove_pedestal, fit_AL, s_d, kafang):
    pedestal_value = AL_getpedestal(path_initial_data, pedestal)
    pedestal.append(pedestal_value)
    # get real ADC value
    y_ADC_remove_pedestal = np.array(y_ADC) - pedestal
    # normalised data
    norm_y_ADC_remove_pedestal = np.array(y_ADC_remove_pedestal)/y_ADC_remove_pedestal[0]
    # fit initial values ADC_0
    ADC_0 = (x_L_L[-1]*y_ADC_remove_pedestal[0] - x_L_L[0]*y_ADC_remove_pedestal[-1])/(x_L_L[-1] - x_L_L[0])
    # fit initial values AL
    estimate_AL = ADC_0*(x_L_L[-1] - x_L_L[0])/(y_ADC_remove_pedestal[0] - y_ADC_remove_pedestal[-1])

    # fit
    popt, pcov = curve_fit(ExpFunc, x_L_L, y_ADC_remove_pedestal, sigma = mean_sigma, p0 = [estimate_AL, ADC_0], bounds=([0, 0], [500, 4096]))

    # get fit data
    fit_ADC_remove_pedestal = []
    for i in x_L_L:
        # get fitting value
        fit_ADC_remove_pedestal_value = ExpFunc(i, popt[0], popt[1])
        # retain three decimal places
        round3_fit_ADC_remove_pedestal_value = np.round(fit_ADC_remove_pedestal_value, 3)
        # write to list
        fit_ADC_remove_pedestal.append(round3_fit_ADC_remove_pedestal_value)
    # normalised fitting data
    norm_fit_ADC_remove_pedestal = np.array(fit_ADC_remove_pedestal)/fit_ADC_remove_pedestal[0]
    
    # get fit AL
    fit_AL_value = round(popt[0], 3)
    fit_AL.append(fit_AL_value)

    # get standard deviation
    s = np.sqrt(np.diag(pcov))
    s_d_value = np.round(s[0], 3)
    s_d.append(s_d_value)

    # get chi^2
    for i, j, k in zip(y_ADC_remove_pedestal, fit_ADC_remove_pedestal, mean_sigma):
        kafang = (((i - j)/k)**2)
        kafang += kafang
        kafang_value = round(kafang, 3)
    kafang.append(kafang_value)
    return pedestal, x_L_L, y_ADC, mean_sigma, norm_y_ADC_remove_pedestal, norm_fit_ADC_remove_pedestal, fit_AL, s_d, kafang

# yourdata
level, ADC, sigma, errorbar, normADC, normfitADC, pedestal, fitAL, sd, kafang = [],[],[],[],[],[],[],[],[],[]
initialpath = Path('your initialpath') 
meandatapath = Path('your meandatapath') 
AL_getmeandata(initialpath, meandatapath)
AL_writemeandata(meandatapath, level, ADC, sigma, errorbar)
data_arrange(initialpath, pedestal, level, ADC, sigma, normADC, normfitADC, fitAL, sd, kafang)


# start plot
fig, ax1 = plt.subplots(figsize = (12, 6))
ax1.set_xlim([0, 1.1])
ax1.set_ylim([0.95, 1.01])
ax1.set_xlabel("Liquid level(m)")
ax1.set_ylabel('Normalization ADC')
ax1.grid(linestyle = '--', color = 'slateblue', alpha = 0.5)

# plot scatter point 
ax1.plot(level, normADC, '.', ms = 6, color='g')


# plot fit curve
ax1.plot(level, normfitADC, linewidth = 2, color='g', alpha = 0.7, label = 'file name')


# other setting
plt.legend(loc = "upper right")
plt.show()


