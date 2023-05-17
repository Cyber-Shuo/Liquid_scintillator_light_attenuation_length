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
    # 获取最后一行的台阶值
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

# 均值数据写入列表函数
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
        # 写入误差棒上下高度
        # y_erroy_bar.append(float((each_y-each_y*(0.9974))/y_ADC[0]))
        y_erroy_bar.append(float(each_y-each_y*(0.9973)))
    return 

# 绘图函数
def LAB_Attenuation_Length_figplot(x_L_L, y_ADC, mean_sigma, y_erroy_bar, path_initial_data, pedestal):
    pedestal = LAB_Attenuation_Length_getpedestal(path_initial_data, pedestal)
    # 得到扣除台阶的真实ADC
    y_ADC_exact_pedestal = np.array(y_ADC) - pedestal
    # 将数据归一化
    norm_y_ADC = np.array(y_ADC_exact_pedestal)/y_ADC_exact_pedestal[0]
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
        round3_x_value_fit_ADC = np.round(x_value_fit_ADC, 3)
        # 写入拟合值列表
        fit_ADC_list.append(round3_x_value_fit_ADC)

    # 获得归一化拟合ADC
    norm_fit_ADC_list = np.array(fit_ADC_list)/fit_ADC_list[0]

    # 得到真实的衰减长度
    Real_AL = round(popt[0], 3)

    # 得到标准差
    s = np.sqrt(np.diag(pcov))
    s_v = np.round(s[0], 3)

    # 计算卡方
    for real_y, fit_y, real_sigma in zip(y_ADC_exact_pedestal, fit_ADC_list, mean_sigma):
        kafang = (((real_y - fit_y)/real_sigma)**2)
        kafang += kafang
        finalkafang = round(kafang, 3)
        # print(finalkafang)
 
    # 创建图片
    fig, ax1 = plt.subplots(figsize = (12, 6))
    # 设置坐标轴
    ax1.set_xlim([0, 1.05])
    ax1.set_ylim([2825,2950])
    x_ticks = np.arange(0, 1.05, 0.05)
    plt.xticks(x_ticks)
    ax1.set_xlabel("Liquid level(m)")
    ax1.set_ylabel('ADC value')
    # 画归一化ADC分布点
    ax1.plot(x_L_L, y_ADC_exact_pedestal, 's', ms = 5, color='black', label = 'ADC value')
    # 画拟合曲线
    ax1.plot(x_L_L, fit_ADC_list, linewidth = 2, color='r', label = 'fit curve')
    # 画出误差棒
    ax1.errorbar(x_L_L, y_ADC_exact_pedestal, yerr = y_erroy_bar, ms = 5, mfc = 'black', mec = 'g', ecolor='g', marker='s', capsize = 3, capthick = 1, linestyle='none')
    # 其他图形设定
    ax1.grid(linestyle = '--', color = 'slateblue', alpha = 0.5)
    plt.legend(loc = "upper right")
    dof = len(x_L_L) - 2
    plt.annotate('AL:{}'.format(Real_AL) + r'$\pm$'+'{}'.format(s_v) + 'm' + '\n' + r'$\chi^{2}/\rm{nDF}$' + '={}'.format(finalkafang) + '/{}'.format(dof), xy=(0.89, 2920), bbox = dict(boxstyle='round', fc='0.9'))
    plt.show()
    return 

pedestal = 0
x_L_L, y_ADC, mean_sigma, y_erroy_bar = [],[],[],[]

single_initial_path = Path('D:\STUDY_think\ALL_Data\\416_LAB-AL-EX-data\ADC-data\\2023-5-2-NJ68-4\\2023-5-2-NJ68-4.txt') 
single_meandata_path = Path('D:\STUDY_think\ALL_Data\\416_LAB-AL-EX-data\meandata\\2023-5-2-NJ68-4.txt') 
LAB_Attenuation_Length_getmeandata(single_initial_path, single_meandata_path)
LAB_Attenuation_Length_write_meandata(single_meandata_path, x_L_L, y_ADC, mean_sigma, y_erroy_bar)
LAB_Attenuation_Length_figplot(x_L_L, y_ADC, mean_sigma, y_erroy_bar, single_initial_path, pedestal)


