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

sns.set(font_scale=1.5)
'''dataset'''
folder_path = "E:/pythonProject/data/COIL_EXAMPLES"

a0 = np.array(Image.open('./data/COIL/obj6__0.png'))
a5 = np.array(Image.open('./data/COIL/obj6__5.png'))
a10 = np.array(Image.open('./data/COIL/obj6__10.png'))
a15 = np.array(Image.open('./data/COIL/obj6__15.png'))
a20 = np.array(Image.open('./data/COIL/obj6__20.png'))
a25 = np.array(Image.open('./data/COIL/obj6__25.png'))
a30 = np.array(Image.open('./data/COIL/obj6__30.png'))
a35 = np.array(Image.open('./data/COIL/obj6__35.png'))
a40 = np.array(Image.open('./data/COIL/obj6__40.png'))
a45 = np.array(Image.open('./data/COIL/obj6__45.png'))

b0 = np.array(Image.open('./data/COIL_SIM/obj25__0.png'))
b5 = np.array(Image.open('./data/COIL_SIM/obj25__5.png'))
b10 = np.array(Image.open('./data/COIL_SIM/obj25__10.png'))
b15 = np.array(Image.open('./data/COIL_SIM/obj25__15.png'))
b20 = np.array(Image.open('./data/COIL_SIM/obj25__20.png'))
b25 = np.array(Image.open('./data/COIL_SIM/obj25__25.png'))
b30 = np.array(Image.open('./data/COIL_SIM/obj25__30.png'))
b35 = np.array(Image.open('./data/COIL_SIM/obj25__35.png'))
b40 = np.array(Image.open('./data/COIL_SIM/obj25__40.png'))
b45 = np.array(Image.open('./data/COIL_SIM/obj25__45.png'))

# 创建一个新的图形
plt.figure()

plt.subplot(2, 10, 1)  # (rows, columns, panel number)
plt.imshow(a0)
plt.axis('off')

plt.subplot(2, 10, 2)  # (rows, columns, panel number)
plt.imshow(a5)
plt.axis('off')

plt.subplot(2, 10, 3)  # (rows, columns, panel number)
plt.imshow(a10)
plt.axis('off')

plt.subplot(2, 10, 4)  # (rows, columns, panel number)
plt.imshow(a15)
plt.axis('off')

plt.subplot(2, 10, 5)  # (rows, columns, panel number)
plt.imshow(a20)
plt.axis('off')

plt.subplot(2, 10, 6)  # (rows, columns, panel number)
plt.imshow(a25)
plt.axis('off')

plt.subplot(2, 10, 7)  # (rows, columns, panel number)
plt.imshow(a30)
plt.axis('off')

plt.subplot(2, 10, 8)  # (rows, columns, panel number)
plt.imshow(a35)
plt.axis('off')

plt.subplot(2, 10, 9)  # (rows, columns, panel number)
plt.imshow(a40)
plt.axis('off')

plt.subplot(2, 10, 10)  # (rows, columns, panel number)
plt.imshow(a45)
plt.axis('off')

plt.subplot(2, 10, 11)  # (rows, columns, panel number)
plt.imshow(b0)
plt.axis('off')

plt.subplot(2, 10, 12)  # (rows, columns, panel number)
plt.imshow(b5)
plt.axis('off')

plt.subplot(2, 10, 13)  # (rows, columns, panel number)
plt.imshow(b10)
plt.axis('off')

plt.subplot(2, 10, 14)  # (rows, columns, panel number)
plt.imshow(b15)
plt.axis('off')

plt.subplot(2, 10, 15)  # (rows, columns, panel number)
plt.imshow(b20)
plt.axis('off')

plt.subplot(2, 10, 16)  # (rows, columns, panel number)
plt.imshow(b25)
plt.axis('off')

plt.subplot(2, 10, 17)  # (rows, columns, panel number)
plt.imshow(b30)
plt.axis('off')

plt.subplot(2, 10, 18)  # (rows, columns, panel number)
plt.imshow(b35)
plt.axis('off')

plt.subplot(2, 10, 19)  # (rows, columns, panel number)
plt.imshow(b40)
plt.axis('off')


plt.subplot(2, 10, 20)  # (rows, columns, panel number)
plt.imshow(b45)
plt.axis('off')
plt.show()
