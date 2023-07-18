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
        # write the height above and below the error bar and normalised to the lowest liquid level ADC
        y_erroy_bar.append(float((each_y-each_y*(0.9974))/y_ADC[0]))
    return 

def data_arrange(path_initial_data, pedestal, x_L_L, y_ADC, mean_sigma, norm_y_ADC, norm_fit_ADC_list, Real_AL, s_v, finalkafang):
    pedestal_value = LAB_Attenuation_Length_getpedestal(path_initial_data, pedestal)
    pedestal.append(pedestal_value)
    # getting to deduct pedestal for real ADCs
    y_ADC_exact_pedestal = np.array(y_ADC) - pedestal
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
    fit_ADC_list = []
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
    return pedestal, x_L_L, y_ADC, mean_sigma, norm_y_ADC, norm_fit_ADC_list, Real_AL, s_v, finalkafang

x_L_L_1, y_ADC_1, mean_sigma_1, y_erroy_bar_1, norm_y_ADC_1, norm_fit_ADC_list_1, pedestal_1, Real_AL_1, s_v_1, finalkafang_1 = [],[],[],[],[],[],[],[],[],[]
single_initial_path_1 = Path('D:\\2021-2024-data\\416_LAB-AL-EX-data\ADC-data\\2021-01-25-NJ64-1\\2021-01-25_NJ_64_1.txt') 
single_meandata_path_1 = Path('D:\\2021-2024-data\\416_LAB-AL-EX-data\meandata\\2021-01-25-NJ64-1.txt') 
LAB_Attenuation_Length_getmeandata(single_initial_path_1, single_meandata_path_1)
LAB_Attenuation_Length_write_meandata(single_meandata_path_1, x_L_L_1, y_ADC_1, mean_sigma_1, y_erroy_bar_1)
data_arrange(single_initial_path_1, pedestal_1, x_L_L_1, y_ADC_1, mean_sigma_1, norm_y_ADC_1, norm_fit_ADC_list_1, Real_AL_1, s_v_1, finalkafang_1)

x_L_L_2, y_ADC_2, mean_sigma_2, y_erroy_bar_2, norm_y_ADC_2, norm_fit_ADC_list_2, pedestal_2, Real_AL_2, s_v_2, finalkafang_2 = [],[],[],[],[],[],[],[],[],[]
single_initial_path_2 = Path('D:\\2021-2024-data\\416_LAB-AL-EX-data\ADC-data\\2021-11-18-NJ66-1\\2021-11-18-NJ-66-31.0+-1.1m.txt') 
single_meandata_path_2 = Path('D:\\2021-2024-data\\416_LAB-AL-EX-data\meandata\\2021-11-18-NJ66-1.txt') 
LAB_Attenuation_Length_getmeandata(single_initial_path_2, single_meandata_path_2)
LAB_Attenuation_Length_write_meandata(single_meandata_path_2, x_L_L_2, y_ADC_2, mean_sigma_2, y_erroy_bar_2)
data_arrange(single_initial_path_2, pedestal_2, x_L_L_2, y_ADC_2, mean_sigma_2, norm_y_ADC_2, norm_fit_ADC_list_2, Real_AL_2, s_v_2, finalkafang_2)

x_L_L_11, y_ADC_11, mean_sigma_11, y_erroy_bar_11, norm_y_ADC_11, norm_fit_ADC_list_11, pedestal_11, Real_AL_11, s_v_11, finalkafang_11 = [],[],[],[],[],[],[],[],[],[]
single_initial_path_11 = Path('D:\\2021-2024-data\\416_LAB-AL-EX-data\ADC-data\\2022-08-09-NJ67\\2022-08-08-NJ67-1.txt') 
single_meandata_path_11 = Path('D:\\2021-2024-data\\416_LAB-AL-EX-data\meandata\\2022-08-08-NJ67-1.txt') 
LAB_Attenuation_Length_getmeandata(single_initial_path_11, single_meandata_path_11)
LAB_Attenuation_Length_write_meandata(single_meandata_path_11, x_L_L_11, y_ADC_11, mean_sigma_11, y_erroy_bar_11)
data_arrange(single_initial_path_11, pedestal_11, x_L_L_11, y_ADC_11, mean_sigma_11, norm_y_ADC_11, norm_fit_ADC_list_11, Real_AL_11, s_v_11, finalkafang_11)

x_L_L_111, y_ADC_111, mean_sigma_111, y_erroy_bar_111, norm_y_ADC_111, norm_fit_ADC_list_111, pedestal_111, Real_AL_111, s_v_111, finalkafang_111 = [],[],[],[],[],[],[],[],[],[]
single_initial_path_111 = Path('D:\\2021-2024-data\\416_LAB-AL-EX-data\ADC-data\\2022-12-05-NJ68\\2022-12-06-NJ68-3.txt') 
single_meandata_path_111 = Path('D:\\2021-2024-data\\416_LAB-AL-EX-data\meandata\\2022-12-06-NJ68-3.txt') 
LAB_Attenuation_Length_getmeandata(single_initial_path_111, single_meandata_path_111)
LAB_Attenuation_Length_write_meandata(single_meandata_path_111, x_L_L_111, y_ADC_111, mean_sigma_111, y_erroy_bar_111)
data_arrange(single_initial_path_111, pedestal_111, x_L_L_111, y_ADC_111, mean_sigma_111, norm_y_ADC_111, norm_fit_ADC_list_111, Real_AL_111, s_v_111, finalkafang_111)


fig, ax1 = plt.subplots(figsize = (9, 6))
ax1.set_xlim([0, 1.1])
ax1.set_ylim([0.95, 1.01])
ax1.set_xlabel("Liquid level(m)")
ax1.set_ylabel('Normalization ADC')
ax1.grid(linestyle = '--', color = 'slateblue', alpha = 0.5)

ax1.plot(x_L_L_1, norm_y_ADC_1, '.', ms = 6, color='g')
ax1.plot(x_L_L_2, norm_y_ADC_2, 'v', ms = 6, color='b')
ax1.plot(x_L_L_11, norm_y_ADC_11, 's', ms = 6, color='y')
ax1.plot(x_L_L_111, norm_y_ADC_111, 'd', ms = 6, color='r')

ax1.plot(x_L_L_1, norm_fit_ADC_list_1, linewidth = 2, color='g', alpha = 0.7, label = 'NJ64')
ax1.plot(x_L_L_2, norm_fit_ADC_list_2, linewidth = 2, color='b', alpha = 0.7, label = 'NJ66')
ax1.plot(x_L_L_11, norm_fit_ADC_list_11, linewidth = 2, color='y', alpha = 0.7, label = 'NJ67')
ax1.plot(x_L_L_111, norm_fit_ADC_list_111, linewidth = 2, color='r', alpha = 0.7, label = 'NJ68')

# ax1.errorbar(x_L_L_1, norm_y_ADC_1, yerr = y_erroy_bar_1, ecolor='g', marker='s', capsize = 3, capthick = 1, linestyle='none')
# ax1.errorbar(x_L_L_2, norm_y_ADC_2, yerr = y_erroy_bar_2, ecolor='b', marker='s', capsize = 3, capthick = 1, linestyle='none')
# ax1.errorbar(x_L_L_11, norm_y_ADC_11, yerr = y_erroy_bar_11, ecolor='y', marker='s', capsize = 3, capthick = 1, linestyle='none')
# ax1.errorbar(x_L_L_111, norm_y_ADC_111, yerr = y_erroy_bar_111, ecolor='r', marker='s', capsize = 3, capthick = 1, linestyle='none')

plt.legend(loc = "upper right")
plt.show()


