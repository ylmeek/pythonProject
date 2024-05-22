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
'''cann_test_acc'''
epochs = np.arange(1, 51)
# le_acc = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_test_cann/LeNet_acc_test.csv', delimiter=',')
#
# alex_acc = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_test_cann/AlexNet_acc_test.csv', delimiter=',')
#
# google_acc = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_test_cann/GoogleNet_acc_test.csv', delimiter=',')
#
# vgg_acc = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_test_cann/VGG16_acc_test.csv', delimiter=',')
#
# res_acc = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_test_cann/ResNet50_acc_test.csv', delimiter=',')
#
# acc1 = [0.6657] * len(epochs)

le_acc = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_test_cann/LeNet_acc_train.csv', delimiter=',')

alex_acc = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_test_cann/AlexNet_acc_train.csv', delimiter=',')

google_acc = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_test_cann/GoogleNet_acc_train.csv', delimiter=',')

vgg_acc = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_test_cann/VGG16_acc_train.csv', delimiter=',')

res_acc = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_test_cann/ResNet50_acc_train.csv', delimiter=',')
acc1 = [0.9895] * len(epochs)

plt.figure(figsize=(10,8))
plt.tick_params(labelsize=25)
Method1, = plt.plot(epochs, acc1, linewidth =2.0,label='BioDet')
# LeNet, = plt.plot(epochs, le_acc, linewidth =2.0,label='LeNet')
AlexNet, = plt.plot(epochs, alex_acc, linewidth =2.0,label='AlexNet')
GoogleNet, = plt.plot(epochs, google_acc, linewidth =2.0,label='GoogleNet')
VGG16, = plt.plot(epochs, vgg_acc, linewidth =2.0,label='VGG16')
ResNet50, = plt.plot(epochs, res_acc, linewidth =2.0,label='ResNet50')
plt.xlabel("The number of epochs", fontsize=30)
plt.ylabel("The accuracy of target orientation", fontsize=30)
plt.legend(fontsize=25)

plt.show()
