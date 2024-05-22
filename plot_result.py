from typing import List

import numpy as np
import pickle
import random
import warnings
from PIL import Image
from tqdm import tqdm
from scipy.io import loadmat
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
#from plottable import ColDef, Table
import seaborn as sns


'''accuracy lifelong'''
# # method1 = np.round(np.loadtxt('./saved_csv/coil_del/pre/longlife_sim_me1.csv', delimiter=',')) #, decimals=1
# # method2 = np.round(np.loadtxt('./saved_csv/coil_del/pre/longlife_sim_me3.csv', delimiter=',') ) #, decimals=1
# # method1 = np.loadtxt('./saved_csv/coil_del/pre/longlife_sim_me1.csv', delimiter=',') #, decimals=1
# # method2 = np.loadtxt('./saved_csv/coil_del/pre/longlife_sim_me3.csv', delimiter=',')  #, decimals=1
# method1 = np.loadtxt('./saved_csv/coil_sim/longlife_sim_me1.csv', delimiter=',') #, decimals=1
# method2 = np.loadtxt('./saved_csv/coil_sim/longlife_sim_me3.csv', delimiter=',')  #, decimals=1
# method1_ = method1[:30, :30]
# method2_ = method2[:30, :30]
# mask = np.zeros_like(method1_)
# mask[np.triu_indices_from(mask)] = True
# plt.figure(figsize=(12,8))
# plt.tick_params(labelsize=25)
# with sns.axes_style("white"):
#     ax = sns.heatmap(method1_, mask=mask, vmax=0.3, annot=True, cmap="RdBu_r", annot_kws={"fontsize":20})
#     #ax.set_title('accuracy on previously trained objects')
#     ax.set_xlabel('object', fontsize=30)  # x轴标题
#     ax.set_ylabel('training times', fontsize=30)
#     plt.show()
#     figure = ax.get_figure()
#     # figure.savefig('./saved_csv/coil_del/longlife_me1.jpg')  # 保存图片
# print('end')

'''memory loss'''
# method1_sim = np.round(np.loadtxt('./saved_csv/coil_sim/longlife_sim_me1.csv', delimiter=','), decimals=3 )
# method3_sim = np.round(np.loadtxt('./saved_csv/coil_sim/longlife_sim_me3.csv', delimiter=','), decimals=3 )
# method1_del = np.round(np.loadtxt('./saved_csv/coil_del/longlife_sim_me1.csv', delimiter=','), decimals=3 )
# method3_del = np.round(np.loadtxt('./saved_csv/coil_del/longlife_sim_me3.csv', delimiter=','), decimals=3 )
#
# last_row1 = method1_del[-1]
# dia1 = np.diag(method1_del)
# result1 = last_row1 - dia1
#
# # last_row2 = method2_del[-1]
# # dia2 = np.diag(method2_del)
# # result2 = last_row2 - dia2
#
# last_row3 = method3_del[-1]
# dia3 = np.diag(method3_del)
# result3 = last_row3 - dia3
#
# last_row4 = method1_sim[-1]
# dia4 = np.diag(method1_sim)
# result4 = last_row4 - dia4
#
# # last_row5 = method2_sim[-1]
# # dia5 = np.diag(method2_sim)
# # result5 = last_row5 - dia5
#
# last_row6 = method3_sim[-1]
# dia6 = np.diag(method3_sim)
# result6 = last_row6 - dia6
#
#
# x = [i for i in range(1, 88)]
# plt.xlabel('Object Number', fontsize=30)  # x轴标题
# plt.ylabel('Memory Loss', fontsize=30)
# plt.plot(result1, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
# # plt.plot(result2, marker='o', markersize=3)
# plt.plot(result3, marker='o', markersize=3)
# plt.plot(result4, marker='o', markersize=3)
# # plt.plot(result5, marker='o', markersize=3)
# plt.plot(result6, marker='o', markersize=3)
# plt.legend(['method1 for dataset1', 'method2 for dataset1',
#             'method1 for dataset2', 'method2 for dataset2'], fontsize=30)  # 设置折线名称
# plt.show()
# print(np.mean(result1)) #method1
# print(np.mean(result4))
#
# print(np.mean(result3))
# print(np.mean(result6))


