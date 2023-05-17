import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import pandas as pd

def ExpFunc(x, AL, ADC_0):
    return ADC_0*np.exp(-x/AL)

def AL_getpedestal(path_initial_data, pedestal):
    Alldata = np.loadtxt(path_initial_data)
    pedestal = Alldata[-1,1]
    return pedestal

def AL_getmeandata(path_initial_data, path_mean_data):
    alldata = np.loadtxt(path_initial_data)
    alldata_remove_pedestal = Alldata[0:-2]
    alldata_remove_pedestal_csv = pd.DataFrame(alldata_remove_pedestal)
    alldata_remove_pedestal_csv_mean = alldata_remove_pedestal_csv.groupby(0).mean()
    meandata = alldata_remove_pedestal_csv_mean.to_csv(path_mean_data, sep='\t', header = False)
    return path_mean_data

def AL_writemeandata(path_mean_data, x_L_L, y_ADC, mean_sigma, y_error_bar):
    with open(path_mean_data,'r') as f:
        lines = f.readlines()
    for line in lines:
        value = [float(s) for s in line.split()]
        x_L_L.append(value[0])
        y_ADC.append(value[1])
        mean_sigma.append(value[2])
    for i in y_ADC:
        y_error_bar.append(float(i-i*(0.9973)))
    return 

def AL_figplot(x_L_L, y_ADC, mean_sigma, y_erroy_bar, path_initial_data, pedestal):
    pedestal = AL_getpedestal(path_initial_data, pedestal)
    y_ADC_remove_pedestal = np.array(y_ADC) - pedestal
    ADC_0 = (x_L_L[-1]*y_ADC_remove_pedestal[0] - x_L_L[0]*y_ADC_remove_pedestal[-1])/(x_L_L[-1] - x_L_L[0])
    estimate_AL = ADC_0*(x_L_L[-1] - x_L_L[0])/(y_ADC_remove_pedestal[0] - y_ADC_remove_pedestal[-1])
    
    popt, pcov = curve_fit(ExpFunc, x_L_L, y_ADC_remove_pedestal, sigma = mean_sigma, p0=[estimate_AL, ADC_0], bounds=([0, 0], [500, 4096]))

    fit_ADC = []
    for iin x_L_L:
        fit_ADC_value = ExpFunc(i, popt[0], popt[1])
        round3_fit_ADC_value = np.round(fit_ADC_value , 3)
        fit_ADC.append(round3_fit_ADC_value)

    fit_AL = round(popt[0], 3)

    s = np.sqrt(np.diag(pcov))
    s_d = np.round(s[0], 3)

    for i, j, k in zip(y_ADC_remove_pedestal, fit_ADC, mean_sigma):
        kafang = (((i - j)/k)**2)
        kafang += kafang
    kafang = round(kafang, 3)

    fig, ax = plt.subplots(figsize = (12, 6))
    ax.set_xlim([0, 1.05])
    ax.set_ylim([2825,2950])
    x_ticks = np.arange(0, 1.05, 0.05)
    plt.xticks(x_ticks)
    ax.set_xlabel("Liquid level(m)")
    ax.set_ylabel('ADC value')

    ax.plot(x_L_L, y_ADC_remove_pedestal, 's', ms = 5, color='black', label = 'ADC value')

    ax.plot(x_L_L, fit_ADC, linewidth = 2, color='r', label = 'fit curve')

    ax.errorbar(x_L_L, y_ADC_remove_pedestal, yerr = y_erroy_bar, ms = 5, mfc = 'black', mec = 'g', ecolor='g', marker='s', capsize = 3, capthick = 1, linestyle='none')

    ax.grid(linestyle = '--', color = 'slateblue', alpha = 0.5)
    plt.legend(loc = "upper right")
    
    dof = len(x_L_L) - 2
    plt.annotate('AL:{}'.format(fit_AL) + r'$\pm$'+'{}'.format(s_d) + 'm' + '\n' + r'$\chi^{2}/\rm{nDF}$' + '={}'.format(kafang) + '/{}'.format(dof), xy=(0.89, 2920), bbox = dict(boxstyle='round', fc='0.9'))
    plt.show()
    return 

pedestal = 0
level, ADC, sigma, errorbar = [],[],[],[]

initialpath = Path('your initialdata') 
meandatapath = Path('your meandata') 
AL_getmeandata(initialpath, meandatapath)
AL_writemeandata(meandatapath, level, ADC, sigma, errorbar)
AL_figplot(level, ADC, sigma, errorbar, initialpath, pedestal)


