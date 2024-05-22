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
from scipy.stats import norm
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
#from plottable import ColDef, Table
import seaborn as sns

sns.set(font_scale=1.5)

sns.set(font_scale=1.5)

true_angles = []
angles = np.loadtxt('./saved_csv/coil_lable_del.csv', dtype=int, delimiter=',')  # , max_rows=144
# lables_all = angles // 5  # sim
for i in range(0, len(angles)):
    if i % 2 != 0:
        true_angles.append(angles[i])

means = np.loadtxt('./saved_csv/cann/means.txt', delimiter=',')  # , max_rows=144
vars = np.loadtxt('./saved_csv/cann/vars.txt', delimiter=',')  # , max_rows=144


plt.boxplot(true_angles)
plt.title('箱形图')
plt.ylabel('值')
# 在图中添加平均数和方差的标记
plt.axhline(y=means, color='r', linestyle='-', label=f'平均数: {means}')
plt.axhline(y=means + np.sqrt(vars), color='g', linestyle='--', label=f'平均数 + 标准差: {means + np.sqrt(vars)}')
plt.axhline(y=means - np.sqrt(vars), color='g', linestyle='--', label=f'平均数 - 标准差: {means - np.sqrt(vars)}')

plt.legend()
plt.show()
epochs = np.arange(0, 3132)
result = 1/(vars*np.sqrt(2*np.pi)) * np.exp(-((true_angles - means) ** 2) / (2 * vars ** 2))
plt.plot(epochs, result, label='BioDet')
plt.show()
print('end')