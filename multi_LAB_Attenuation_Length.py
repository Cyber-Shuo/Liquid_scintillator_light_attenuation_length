from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import string
from scipy.optimize import curve_fit
import pandas as pd



# 定义拟合函数
def ExpFunc(x, AL, ADC_0):
    return ADC_0*np.exp(-x/AL)

# 获取台阶值函数
def LAB_Attenuation_Length_getpedestal(path_initial_data, pedestal):
    # 载入已经高斯拟合的数据
    Alldata = np.loadtxt(path_initial_data)
    # 获取最后一行的台阶值。
    pedestal = Alldata[-1,1]
    return pedestal

# 平均值数据获取函数
def LAB_Attenuation_Length_getmeandata(path_initial_data, path_mean_data):
    # 载入已经高斯拟合的数据
    Alldata = np.loadtxt(path_initial_data)
    # 将最后一行pedestal数据去除
    Alldata_exact_pedestal = Alldata[0:-2]
    # print(Alldata_exact_pedestal)
    # 转为表格数据
    Excel_Alldata_exact_pedestal = pd.DataFrame(Alldata_exact_pedestal)
    # print(Excel_Alldata_exact_pedestal)
    # 以第一列数据为组取平均值得到表格数据
    Excel_Alldata_exact_pedestal_mean = Excel_Alldata_exact_pedestal.groupby(0).mean()
    # print(Excel_Alldata_exact_pedestal_mean)
    # 将处理好的excel数据保存为txt文件
    meandata = Excel_Alldata_exact_pedestal_mean.to_csv(path_mean_data, sep='\t', header = False)
    return path_mean_data

# 数据写入列表函数
def LAB_Attenuation_Length_write_meandata(path_mean_data, x_L_L, y_ADC, mean_sigma, y_erroy_bar):
    # 打开平均值文件按行写入
    with open(path_mean_data,'r') as f:
        lines = f.readlines()
        # linesdata = lines[0:-1]
        # print(lines)
    # 将液位、ADC、误差数据分别写入3个列表
    for line in lines:
        value = [float(s) for s in line.split()]
        x_L_L.append(value[0])
        y_ADC.append(value[1])
        mean_sigma.append(value[2])
    # 获取误差棒数据
    for each_y in y_ADC:
        # 写入误差棒上下高度 以最低液面ADC为归一化
        y_erroy_bar.append(float((each_y-each_y*(0.9974))/y_ADC[0]))
    return 

def data_arrange(path_initial_data, pedestal, x_L_L, y_ADC, mean_sigma, norm_y_ADC, norm_fit_ADC_list, Real_AL, s_v, finalkafang):
    pedestal_value = LAB_Attenuation_Length_getpedestal(path_initial_data, pedestal)
    pedestal.append(pedestal_value)
    # 得到扣除台阶的真实ADC
    y_ADC_exact_pedestal = np.array(y_ADC) - pedestal
    # 将数据归一化
    norm_y_ADC_value = np.array(y_ADC_exact_pedestal)/y_ADC_exact_pedestal[0]
    # print(norm_y_ADC_value)
    for i in norm_y_ADC_value:
        norm_y_ADC.append(i)
    # 泰勒展开到一阶预估初始液位ADC
    initial_L_ADC = (x_L_L[-1]*y_ADC_exact_pedestal[0] - x_L_L[0]*y_ADC_exact_pedestal[-1])/(x_L_L[-1] - x_L_L[0])
    # 再用预估的初始液位ADC预估衰减长度
    estimate_AL = initial_L_ADC*(x_L_L[-1] - x_L_L[0])/(y_ADC_exact_pedestal[0] - y_ADC_exact_pedestal[-1])

    # 用上文定义指数拟合函数去拟合
    popt, pcov = curve_fit(ExpFunc, x_L_L, y_ADC_exact_pedestal, sigma = mean_sigma, p0=[estimate_AL, initial_L_ADC], bounds=([0, 0], [100, 4096]))

    # 得到拟合ADC的列表
    fit_ADC_list = []
    for x_value in x_L_L:
        # 得到拟合ADC的值
        x_value_fit_ADC = ExpFunc(x_value, popt[0], popt[1])
        # 拟合值取3位有效数字
        round3_x_value_fit_ADC = np.round(x_value_fit_ADC, 3)
        # 写入拟合值列表
        fit_ADC_list.append(round3_x_value_fit_ADC)

    # 获得归一化拟合ADC
    norm_fit_ADC_list_value = np.array(fit_ADC_list)/fit_ADC_list[0]
    for i in norm_fit_ADC_list_value:
        norm_fit_ADC_list.append(i)
    # 得到真实的衰减长度
    Real_AL_value = round(popt[0], 3)
    Real_AL.append(Real_AL_value)

    # 得到标准差
    s = np.sqrt(np.diag(pcov))
    s_v_value = np.round(s[0], 3)
    s_v.append(s_v_value)

    # 计算卡方
    for real_y, fit_y, real_sigma in zip(y_ADC_exact_pedestal, fit_ADC_list, mean_sigma):
        kafang = (((real_y - fit_y)/real_sigma)**2)
        kafang += kafang
        finalkafang_value = round(kafang, 3)
        # print(finalkafang)
    finalkafang.append(finalkafang_value)
    return pedestal, x_L_L, y_ADC, mean_sigma, norm_y_ADC, norm_fit_ADC_list, Real_AL, s_v, finalkafang

x_L_L_1, y_ADC_1, mean_sigma_1, y_erroy_bar_1, norm_y_ADC_1, norm_fit_ADC_list_1, pedestal_1, Real_AL_1, s_v_1, finalkafang_1 = [],[],[],[],[],[],[],[],[],[]
single_initial_path_1 = Path('D:\STUDY_think\ALL_Data\\416_LAB-AL-EX-data\ADC-data\\2022-8-9-NJ67\\2022-08-08-NJ67-1.txt') 
single_meandata_path_1 = Path('D:\STUDY_think\ALL_Data\\416_LAB-AL-EX-data\meandata\\2022-08-08-NJ67-1.txt') 
LAB_Attenuation_Length_getmeandata(single_initial_path_1, single_meandata_path_1)
LAB_Attenuation_Length_write_meandata(single_meandata_path_1, x_L_L_1, y_ADC_1, mean_sigma_1, y_erroy_bar_1)
data_arrange(single_initial_path_1, pedestal_1, x_L_L_1, y_ADC_1, mean_sigma_1, norm_y_ADC_1, norm_fit_ADC_list_1, Real_AL_1, s_v_1, finalkafang_1)

x_L_L_2, y_ADC_2, mean_sigma_2, y_erroy_bar_2, norm_y_ADC_2, norm_fit_ADC_list_2, pedestal_2, Real_AL_2, s_v_2, finalkafang_2 = [],[],[],[],[],[],[],[],[],[]
single_initial_path_2 = Path('D:\STUDY_think\ALL_Data\\416_LAB-AL-EX-data\ADC-data\\2022-8-9-NJ67\\2022-08-08-NJ67-2.txt') 
single_meandata_path_2 = Path('D:\STUDY_think\ALL_Data\\416_LAB-AL-EX-data\meandata\\2022-08-08-NJ67-2.txt') 
LAB_Attenuation_Length_getmeandata(single_initial_path_2, single_meandata_path_2)
LAB_Attenuation_Length_write_meandata(single_meandata_path_2, x_L_L_2, y_ADC_2, mean_sigma_2, y_erroy_bar_2)
data_arrange(single_initial_path_2, pedestal_2, x_L_L_2, y_ADC_2, mean_sigma_2, norm_y_ADC_2, norm_fit_ADC_list_2, Real_AL_2, s_v_2, finalkafang_2)

x_L_L_11, y_ADC_11, mean_sigma_11, y_erroy_bar_11, norm_y_ADC_11, norm_fit_ADC_list_11, pedestal_11, Real_AL_11, s_v_11, finalkafang_11 = [],[],[],[],[],[],[],[],[],[]
single_initial_path_11 = Path('D:\STUDY_think\ALL_Data\\416_LAB-AL-EX-data\ADC-data\\2022-12-5-NJ68\\2022-12-06-NJ68-4.txt') 
single_meandata_path_11 = Path('D:\STUDY_think\ALL_Data\\416_LAB-AL-EX-data\meandata\\2022-12-06-NJ68-4.txt') 
LAB_Attenuation_Length_getmeandata(single_initial_path_11, single_meandata_path_11)
LAB_Attenuation_Length_write_meandata(single_meandata_path_11, x_L_L_11, y_ADC_11, mean_sigma_11, y_erroy_bar_11)
data_arrange(single_initial_path_11, pedestal_11, x_L_L_11, y_ADC_11, mean_sigma_11, norm_y_ADC_11, norm_fit_ADC_list_11, Real_AL_11, s_v_11, finalkafang_11)

x_L_L_111, y_ADC_111, mean_sigma_111, y_erroy_bar_111, norm_y_ADC_111, norm_fit_ADC_list_111, pedestal_111, Real_AL_111, s_v_111, finalkafang_111 = [],[],[],[],[],[],[],[],[],[]
single_initial_path_111 = Path('D:\STUDY_think\ALL_Data\\416_LAB-AL-EX-data\ADC-data\\2022-12-5-NJ68\\2022-12-05-NJ68-1.txt') 
single_meandata_path_111 = Path('D:\STUDY_think\ALL_Data\\416_LAB-AL-EX-data\meandata\\2022-12-05-NJ68-1.txt') 
LAB_Attenuation_Length_getmeandata(single_initial_path_111, single_meandata_path_111)
LAB_Attenuation_Length_write_meandata(single_meandata_path_111, x_L_L_111, y_ADC_111, mean_sigma_111, y_erroy_bar_111)
data_arrange(single_initial_path_111, pedestal_111, x_L_L_111, y_ADC_111, mean_sigma_111, norm_y_ADC_111, norm_fit_ADC_list_111, Real_AL_111, s_v_111, finalkafang_111)


# 创建图片
fig, ax1 = plt.subplots(figsize = (12, 6))
ax1.set_xlim([0, 1.1])
ax1.set_ylim([0.95, 1.01])
# 设置坐标轴标注
ax1.set_xlabel("Liquid level(m)")
ax1.set_ylabel('Normalization ADC')
ax1.grid(linestyle = '--', color = 'slateblue', alpha = 0.5)

# 画归一化ADC分布点
# ax1.plot(0.1, 1.007, '.', ms = 6, color='black', label = 'Normalization ADC value')
ax1.plot(x_L_L_1, norm_y_ADC_1, '.', ms = 6, color='g')
ax1.plot(x_L_L_2, norm_y_ADC_2, 'v', ms = 6, color='b')
ax1.plot(x_L_L_11, norm_y_ADC_11, 's', ms = 6, color='y')
ax1.plot(x_L_L_111, norm_y_ADC_111, 'd', ms = 6, color='r')
# 画拟合曲线
# ax1.hlines(y = 1.005, xmin = 0.05, xmax = 0.15, color = 'black', label = 'fit curve')
ax1.plot(x_L_L_1, norm_fit_ADC_list_1, linewidth = 2, color='g', alpha = 0.7, label = 'NJ67-1')
ax1.plot(x_L_L_2, norm_fit_ADC_list_2, linewidth = 2, color='b', alpha = 0.7, label = 'NJ67-2')
ax1.plot(x_L_L_11, norm_fit_ADC_list_11, linewidth = 2, color='y', alpha = 0.7, label = 'NJ68-4')
ax1.plot(x_L_L_111, norm_fit_ADC_list_111, linewidth = 2, color='r', alpha = 0.7, label = 'NJ68-1')
# 画出误差棒
# ax1.errorbar(x_L_L_1, norm_y_ADC_1, yerr = y_erroy_bar_1, ecolor='g', marker='s', capsize = 3, capthick = 1, linestyle='none')
# ax1.errorbar(x_L_L_2, norm_y_ADC_2, yerr = y_erroy_bar_2, ecolor='b', marker='s', capsize = 3, capthick = 1, linestyle='none')
# ax1.errorbar(x_L_L_11, norm_y_ADC_11, yerr = y_erroy_bar_11, ecolor='y', marker='s', capsize = 3, capthick = 1, linestyle='none')
# ax1.errorbar(x_L_L_111, norm_y_ADC_111, yerr = y_erroy_bar_111, ecolor='r', marker='s', capsize = 3, capthick = 1, linestyle='none')

# 其他图形设定
plt.legend(loc = "upper right")

# 设置AL标注
# plt.annotate('AL:{}'.format(Real_AL_1[0]) + r'$\pm$'+'{}'.format(s_v_1[0]) + 'm' + '\n' + 'AL:{}'.format(Real_AL_2[0]) + r'$\pm$'+'{}'.format(s_v_2[0]) + 'm' + '\n' +'AL:{}'.format(Real_AL_11[0]) + r'$\pm$'+'{}'.format(s_v_11[0]) + 'm' + '\n' +'AL:{}'.format(Real_AL_111[0]) + r'$\pm$'+'{}'.format(s_v_111[0]) + 'm' , xy=(0.8, 0.995), bbox = dict(boxstyle='round', fc='0.8'))
plt.show()


