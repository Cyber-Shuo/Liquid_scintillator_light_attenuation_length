from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import pandas as pd

# define the fitting function
def ExpFunc(x, AL, ADC_0):
    return ADC_0*np.exp(-x/AL)

# get pedestal function
def LAB_Attenuation_Length_getpedestal(path_initial_data, pedestal):
    # load data that has been Gaussian fitted
    Alldata = np.loadtxt(path_initial_data)
    # gets the pedestal value of the last row
    pedestal = Alldata[-1,1]
    return pedestal

# mean value data acquisition function
def LAB_Attenuation_Length_getmeandata(path_initial_data, path_mean_data):
    # load data that has been Gaussian fitted
    Alldata = np.loadtxt(path_initial_data)
    # remove the last line of pedestal data
    Alldata_exact_pedestal = Alldata[0:-1]
    # conversion to csv data
    Excel_Alldata_exact_pedestal = pd.DataFrame(Alldata_exact_pedestal)
    # take the first column of data as a group and take the average to get the csv data
    Excel_Alldata_exact_pedestal_mean = Excel_Alldata_exact_pedestal.groupby(0).mean()
    # save processed csv data as a txt file
    meandata = Excel_Alldata_exact_pedestal_mean.to_csv(path_mean_data, sep='\t', header = False)
    return path_mean_data

# write data to list functions
def LAB_Attenuation_Length_write_meandata(path_mean_data, x_L_L, y_ADC, mean_sigma, y_erroy_bar):
    # open average file and write by line
    with open(path_mean_data,'r') as f:
        lines = f.readlines()
    for line in lines:
        value = [float(s) for s in line.split()]
        x_L_L.append(value[0])
        y_ADC.append(value[1])
        mean_sigma.append(value[2])
    for each_y in y_ADC:
        # write the height above and below the error bar
        y_erroy_bar.append(float((each_y-each_y*(0.9974))))
    return 

def data_arrange(path_initial_data, pedestal, x_L_L, y_ADC, mean_sigma, norm_y_ADC, norm_fit_ADC_list, Real_AL, s_v, finalkafang, fit_ADC_list, y_ADC_exact_pedestal):
    pedestal_value = LAB_Attenuation_Length_getpedestal(path_initial_data, pedestal)
    pedestal.append(pedestal_value)
    # getting to deduct pedestal for real ADCs
    y_ADC_exact_pedestal_list = np.array(y_ADC) - pedestal
    for i in y_ADC_exact_pedestal_list:
        y_ADC_exact_pedestal.append(i)
    # normalised to the lowest liquid level ADC
    norm_y_ADC_value = np.array(y_ADC_exact_pedestal)/y_ADC_exact_pedestal[0]
    for i in norm_y_ADC_value:
        norm_y_ADC.append(i)
    # Taylor Expansion to first order predictive initial level ADCs
    initial_L_ADC = (x_L_L[-1]*y_ADC_exact_pedestal[0] - x_L_L[0]*y_ADC_exact_pedestal[-1])/(x_L_L[-1] - x_L_L[0])
    # then the predicted initial level ADC is used to predict the attenuation length
    estimate_AL = initial_L_ADC*(x_L_L[-1] - x_L_L[0])/(y_ADC_exact_pedestal[0] - y_ADC_exact_pedestal[-1])

    # the exponential fitting function defined above was used to fit
    popt, pcov = curve_fit(ExpFunc, x_L_L, y_ADC_exact_pedestal, sigma = mean_sigma, p0=[estimate_AL, initial_L_ADC], bounds=([0, 0], [100, 4096]))

    # get the list of fitted ADCs
    for x_value in x_L_L:
        x_value_fit_ADC = ExpFunc(x_value, popt[0], popt[1])
        round2_x_value_fit_ADC = np.round(x_value_fit_ADC, 2)
        fit_ADC_list.append(round2_x_value_fit_ADC)

    # normalised to the lowest liquid level fitted ADC
    norm_fit_ADC_list_value = np.array(fit_ADC_list)/fit_ADC_list[0]
    for i in norm_fit_ADC_list_value:
        norm_fit_ADC_list.append(i)
    # get the true attenuation length
    Real_AL_value = round(popt[0], 2)
    Real_AL.append(Real_AL_value)

    # get the standard deviation
    s = np.sqrt(np.diag(pcov))
    s_v_value = np.round(s[0], 2)
    s_v.append(s_v_value)

    # calculate the chi-square
    for real_y, fit_y, real_sigma in zip(y_ADC_exact_pedestal, fit_ADC_list, mean_sigma):
        kafang = (((real_y - fit_y)/real_sigma)**2)
        kafang += kafang
        finalkafang_value = round(kafang, 3)
    finalkafang.append(finalkafang_value)
    return pedestal, x_L_L, y_ADC, mean_sigma, norm_y_ADC, norm_fit_ADC_list, Real_AL, s_v, finalkafang, fit_ADC_list, y_ADC_exact_pedestal

x_L_L, y_ADC, mean_sigma, y_erroy_bar, norm_y_ADC, norm_fit_ADC_list, pedestal, Real_AL, s_v, finalkafang, fit_ADC_list, y_ADC_exact_pedestal = [],[],[],[],[],[],[],[],[],[],[],[]
single_initial_path = Path('D:\\2021-2024-data\\416_LAB-AL-EX-data\ADC-data\\2022-12-05-NJ68\\2022-12-06-NJ68-3.txt') 
single_meandata_path = Path('D:\\2021-2024-data\\416_LAB-AL-EX-data\meandata\\2022-12-06-NJ68-3.txt') 
LAB_Attenuation_Length_getmeandata(single_initial_path, single_meandata_path)
LAB_Attenuation_Length_write_meandata(single_meandata_path, x_L_L, y_ADC, mean_sigma, y_erroy_bar)
data_arrange(single_initial_path, pedestal, x_L_L, y_ADC, mean_sigma, norm_y_ADC, norm_fit_ADC_list, Real_AL, s_v, finalkafang, fit_ADC_list, y_ADC_exact_pedestal)

fig, ax1 = plt.subplots(figsize = (9, 6))
ax1.set_xlim([0, 1.05])
ax1.set_ylim([2950,3100])
x_ticks = np.arange(0, 1.05, 0.05)
plt.xticks(x_ticks)
ax1.set_xlabel("Liquid level(m)")
ax1.set_ylabel('ADC value')

ax1.plot(x_L_L, y_ADC_exact_pedestal, 's', ms = 5, color='black', label = 'ADC value')

ax1.plot(x_L_L, fit_ADC_list, linewidth = 2, color='r', label = 'fit curve')

ax1.errorbar(x_L_L, y_ADC_exact_pedestal, yerr = y_erroy_bar, ms = 5, mfc = 'black', mec = 'g', ecolor='g', marker='s', capsize = 3, capthick = 1,linestyle='none')

ax1.grid(linestyle = '--', color = 'slateblue', alpha = 0.5)

plt.legend(loc = "upper right")
dof = len(x_L_L) - 2
plt.annotate('AL:{}'.format(Real_AL[0]) + r'$\pm$'+'{}'.format(s_v[0]) + 'm' + '\n' + r'$\chi^{2}/\rm{nDF}$' + '={}'.format(finalkafang[0]) + '/{}'.format(dof), xy=(0.85, 3070), bbox = dict(boxstyle='round', fc='0.9'))
plt.show()
