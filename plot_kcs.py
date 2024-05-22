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

sns.set(font_scale=1.5)

'''Active KCs'''
kc = [2*20, 32*20, 64*20, 128*20, 256*20, 512*20]
acc_method1_data1 = [0.0410281, 0.07583014, 0.31689017, 0.63793103, 0.84179438, 0.92927842]
acc_method2_data1 = [0.01388889,  0.01851852, 0.13521711, 0.48387612, 0.77426564, 0.91251596]

acc_method1_data2 = [0.01923077, 0.37820513, 0.73931624, 0.8792735, 0.9508547, 0.97649573]
acc_method2_data2 = [0.01388889, 0.35897436, 0.71794872, 0.84401709, 0.9465812, 0.97863248]

plt.figure(figsize=(10,8))
plt.tick_params(labelsize=25)
plt.plot(kc, acc_method1_data1, label='method1 on COIL-100-O')
plt.plot(kc, acc_method2_data1, label='method2 on COIL-100-O')
plt.plot(kc, acc_method1_data2, label='method1 on COIL-100-AS')
plt.plot(kc, acc_method2_data2, label='method2 on COIL-100-AS')
plt.xlabel("The number of active KCs", fontsize=30)
plt.ylabel("The accuracy of target MBON", fontsize=30)
plt.legend(fontsize=25)
plt.show()