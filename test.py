
from cProfile import label
from turtle import color
from matplotlib import offsetbox
import numpy as np
import matplotlib.pyplot as plt
import PyEMD
from scipy.stats import pearsonr

data_year = np.array([
    2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020
    ],)

data_irrigation = np.array([
    240.30,270.10,243.30,247.80,237.20,242.80,259.50,264.10,267.80,273.80,270.30,266.20,255.20,238.70,244.30,239.60,264.40 
    ],)

data_precipitation = np.array([
    1236.00,798.50,1088.10,1006.80,1410.50,1257.10,1044.50,833.40,953.90,1012.10,989.50,1031.70,944.30,1089.00,1021.20,1084.00,784.30 
    ],)


data_irrigation = data_irrigation[::-1]
data_precipitation = data_precipitation[::-1]
E_IMFs = PyEMD.EMD()
imfs = E_IMFs.emd(data_irrigation)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

config = {
    "font.family": "serif",
    "font.size": 10,
    "mathtext.fontset": "stix",
    "font.serif": ["SimHei"]
}
plt.rcParams.update(config)
plt.rcParams['axes.unicode_minus'] = False

fig, axs = plt.subplots(4, 1, sharex=True, dpi=500)
axs[0].plot(data_year, data_irrigation, color="black", linewidth=0.8, label="original data ($\\times10^8m^3$)")
# axs[0].legend(edgecolor='w',fontsize=9,loc=0)
axs[0].set_ylabel("灌溉总量\n$\\times10^8m^3$")

axs[1].plot(data_year, imfs[0], color="r", linewidth=0.8, label="imf1 data ($\\times10^8m^3$)")
# axs[1].legend(edgecolor='w',fontsize=9,loc=0)
axs[1].spines['top'].set_color('r')
axs[1].spines['bottom'].set_color('r')
axs[1].spines['right'].set_color('r')
axs[1].spines['left'].set_color('r')
axs[1].set_ylabel("$\mathrm{imf1}$\n$\\times10^8m^3$")

axs[2].plot(data_year, imfs[1], color="r", linewidth=0.8, label="imf1 data ($\\times10^8m^3$)")
# axs[2].legend(edgecolor='w',fontsize=9,loc=0)
axs[2].spines['top'].set_color('r')
axs[2].spines['bottom'].set_color('r')
axs[2].spines['right'].set_color('r')
axs[2].spines['left'].set_color('r')
axs[2].set_ylabel("$\mathrm{imf2}$\n$\\times10^8m^3$")

axs[3].plot(data_year, imfs[2], color="r", linewidth=0.8, label="imf1 data ($\\times10^8m^3$)")
# axs[3].legend(edgecolor='w',fontsize=9,loc=0)
axs[3].spines['top'].set_color('r')
axs[3].spines['bottom'].set_color('r')
axs[3].spines['right'].set_color('r')
axs[3].spines['left'].set_color('r')
axs[3].set_ylabel("余波\n$\\times10^8m^3$")
axs[3].set_xlabel("年份")
# fig.subplots_adjust(hspace=0)


fig = plt.figure(figsize=(5,8), dpi=500, )
ax1 = fig.add_subplot(4,1,1)
ax1.plot(data_year, data_irrigation, color="black", label="original data ($\\times10^8m^3$)")
ax1.legend(edgecolor='w',fontsize=9,loc=0)

ax2 = fig.add_subplot(4,1,2)
ax2.plot(data_year, imfs[0], color="black", label="imf1 data ($\\times10^8m^3$)")
ax2.legend(edgecolor='w',fontsize=9,loc=0)

ax3 = fig.add_subplot(4,1,3)
ax3.plot(data_year, imfs[1], color="black", label="imf1 data ($\\times10^8m^3$)")
ax3.legend(edgecolor='w',fontsize=9,loc=0)

ax4 = fig.add_subplot(4,1,4)
ax4.plot(data_year, imfs[2], color="black", label="imf1 data ($\\times10^8m^3$)")
ax4.legend(edgecolor='w',fontsize=9,loc=0)
plt.tight_layout()








x = np.random.random((10,))

plt.plot(x,label='随机数')
plt.title('中文：宋体 \n 英文：$\mathrm{Times \; New \; Roman}$ \n 公式： $a_i + \\beta_i = \\gamma^k$')
plt.xlabel('横坐标')
plt.ylabel('纵坐标')
plt.legend()
plt.yticks(fontproperties='Times New Roman', size=18)
plt.xticks(fontproperties='Times New Roman', size=18)
plt.show()



## read meteorological data
import pandas  as pd
from sklearn.linear_model import LinearRegression
from matplotlib.offsetbox import AnnotationBbox, TextArea

def normal_mean(data, index_year, name, bad_value_up=200, bad_value_done=-50):
    """
    calculate the mean value of one weather element, which could be determined by the parameter "name", 
    from 2000 to 2020 based on the 12 stations data.  
    there is some invalid value in original data, the invalid value always large, so we use a range to 
    judge the valid value.

    parameter:
            data >> the original data, we read the data from the path "F:/水博士/开学前/江苏农业灌溉水/
                    数据收集/主要数据.xlsx"
            
            index_year >> the year list (2000-2020) 
            
            name >> the name of weather element, here is some we can choose: "平均气温(℃)", "平均最低气
                    温(℃)", "平均最高气温(℃)", "平均2分钟风速(m/s)", "平均相对湿度(%)", "平均水气压(hPa)"
                     and so on
            
            bad_value_up, bad_value_done >> the range for valid valeu

    return:
            data_mean >> there are 3 series for return, which are the mean value, the year, the number of 
                        valid value
    """
    data_mean = [];
    for year in index_year:
        data_year = np.array(data[data["年"] == year][name])
        data_bool = np.where(((data_year<bad_value_up) & (data_year>bad_value_done)), True, False)
        valid_value = len(data_year[data_bool])
        data_mean.append((np.mean(data_year[data_bool]), year, valid_value))
    return np.array(data_mean)

def line_model(X, y):
    """
    
    """
    model = LinearRegression().fit(X,y)



path_meteorology = "F:/水博士/开学前/江苏农业灌溉水/数据收集/主要数据.xlsx"

data_meteorology = pd.read_excel(path_meteorology, sheet_name="气象数据（12站）",header=0)

data_year = data_meteorology["年"].unique()
data_name = ["平均气温(℃)", "平均最低气温(℃)", "平均最高气温(℃)", "平均2分钟风速(m/s)", "平均相对湿度(%)", "平均水气压(hPa)"]
data_sum = np.zeros((21, 7), dtype=float)
for i, name in enumerate(data_name):
    data_tem = normal_mean(data_meteorology, data_year, name)
    data_sum[:, i+1] = data_tem[:, 0]
data_sum[4:, 0] = data_precipitation
data_foranalyze = data_sum[4:,:]

data_corr = np.zeros((3, 7))
data_corr_p = np.zeros((3, 7))
for i, imf in enumerate(imfs):
    for j in range(data_foranalyze.shape[1]):
        data_corr[i,j], data_corr_p[i,j] = pearsonr(imf, data_foranalyze[:,j])


config = {
    "font.family": "serif",
    "font.size": 10,
    "mathtext.fontset": "stix",
    "font.serif": ["SimHei"]
}
plt.rcParams.update(config)
plt.rcParams['axes.unicode_minus'] = False

row_label = ["$\\mathrm{imf1}$", "$\\mathrm{imf2}$", "$\\mathrm{other}$"]
col_label = ["$\\mathrm{P}$","$\\mathrm{T}_{mean}$", "$\\mathrm{T}_{max}$", "$\\mathrm{T}_{min}$", "$\\mathrm{W_s}$", "$\\mathrm{H}$", "$\\mathrm{V_p}$"]

fig, ax = plt.subplots(dpi=500)
im = ax.imshow(data_corr, cmap="tab20b")

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("$\\mathrm{Correlation}$", rotation=-90, va="bottom")
cbar.ax.set_yticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
cbar.ax.set_yticklabels(["$\\mathrm{-0.6}$", "$\\mathrm{-0.4}$", "$\\mathrm{-0.2}$", " $\\mathrm{0.0}$", " $\\mathrm{0.2}$", " $ \\mathrm{0.4}$", " $ \\mathrm{0.6}$",])

ax.set_xticks(np.arange(len(col_label)))
ax.set_xticklabels(col_label)
ax.set_yticks(np.arange(len(row_label)))
ax.set_yticklabels(row_label)
for i in range(len(row_label)):
    for j in range(len(col_label)):
        if data_corr_p[i, j] <= 0.01:
            text = "$\\mathrm{" + "{:.2f}".format(data_corr[i, j]) + "**}$"
        elif data_corr_p[i, j] <= 0.05:
            text = "$\\mathrm{" + "{:.2f}".format(data_corr[i, j]) + "*}$"
        else:
            text = "$\\mathrm{" + "{:.2f}".format(data_corr[i, j]) +"}$"
        text = ax.text(j, i, text, ha="center", va="center", color="w")

# text = '''$\\mathrm{P: Precipitation (mm); T_{mean} : Temperature (^{\\circ}C);
#             T_{max} : Maximum Temperature (^{\\circ}C); T_{min} : Minimum Temperature (^{\\circ}C);
#             W_s : Wind Spend (m/s);  H : Relative Humidity;
#             V_p : Water Vapor Pressure$'''
text = TextArea(
'''$\\mathrm{P:\:Precipitation (mm),}$ $\\mathrm{T_{mean}:\:Temperature (^{\\circ}C),}$ $\\mathrm{T_{max} :\:Max-Temperature (^{\\circ}C),}$
$\\mathrm{T_{min}: Min-Temperature (^{\\circ}C),}$ $\\mathrm{H:\:Relative\:Humidity,}$ $\\mathrm{W_s:\:Wind\:Spend (m/s),}$
$\\mathrm{V_p:\:Water\:Vapor\:Pressure(hPa)}$
$\\mathrm{Note:}$ $\\mathrm{'*'\:means\:p\:<\:0.05,\:'**'\:means\:p\:<\:0.01}$''')
ab = AnnotationBbox(text, (0.43,0.09), xycoords='figure fraction', fontsize=2, bboxprops =dict(edgecolor='w'))
ax.add_artist(ab)

#plt.subplots_adjust(left=0.1, bottom=0.1, right=None, wspace=None, hspace=None)
fig.tight_layout()

























data_temperture_mean = np.array(data_meteorology["平均气温(℃)"])
data_temperture_minmean = np.array(data_meteorology["平均最低气温(℃)"])
data_temperture_maxmean = np.array(data_meteorology["平均最高气温(℃)"])
data_windspeed = np.array(data_meteorology["平均2分钟风速(m/s)"])
data_humdity = np.array(data_meteorology["平均相对湿度(%)"])
data_vaporpresure = np.array(data_meteorology["平均水气压(hPa)"])





