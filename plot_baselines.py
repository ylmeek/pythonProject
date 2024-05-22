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

'''baseline acc'''
# # epochs1 = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil_del/LeNet_epoch.csv', delimiter=',')
# # le_acc = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil_del/LeNet_acc_train.csv', delimiter=',')
# # le_loss = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil_del/LeNet_loss_train.csv', delimiter=',')
#
# epochs2 = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil_del/AlexNet_epoch.csv', delimiter=',')
# alex_acc = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil_del/AlexNet_acc_train.csv', delimiter=',')
# alex_loss = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil_del/AlexNet_loss_train.csv', delimiter=',')
#
# epochs3 = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil_del/GoogleNet_epoch.csv', delimiter=',')
# google_acc = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil_del/GoogleNet_acc_train.csv', delimiter=',')
# google_loss = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil_del/GoogleNet_loss_train.csv', delimiter=',')
#
# epochs4 = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil_del/VGG16_epoch.csv', delimiter=',')
# vgg_acc = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil_del/VGG16_acc_train.csv', delimiter=',')
# vgg_loss = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil_del/VGG16_loss_train.csv', delimiter=',')
#
# epochs5 = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil_del/ResNet50_epoch.csv', delimiter=',')
# res_acc = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil_del/ResNet50_acc_train.csv', delimiter=',')
# res_loss = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil_del/ResNet50_loss_train.csv', delimiter=',')
#
# acc1 = [0.92927842] * len(epochs3)
# acc2 = [0.91251596] * len(epochs3)
epochs = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil100_sim/coil_epoch.csv', delimiter=',')

le_acc = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil100_sim/LeNet_acc.csv', delimiter=',')
le_loss = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil100_sim/LeNet_loss.csv', delimiter=',')

alex_acc = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil100_sim/AlexNet_acc.csv', delimiter=',')
alex_loss = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil100_sim/AlexNet_loss.csv', delimiter=',')

google_acc = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil100_sim/GoogleNet_acc.csv', delimiter=',')
google_loss = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil100_sim/GoogleNet_loss.csv', delimiter=',')

vgg_acc = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil100_sim/VGG16_acc.csv', delimiter=',')
vgg_loss = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil100_sim/VGG16_loss.csv', delimiter=',')

res_acc = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil100_sim/ResNet50_acc.csv', delimiter=',')
res_loss = np.loadtxt('E:/Image_Classfiation_Coil20-master/result/train_notest/coil100_sim/ResNet50_loss.csv', delimiter=',')

acc1 = [0.97649573] * len(epochs)
acc2 = [0.97863248] * len(epochs)
#
plt.figure(figsize=(10,8))
plt.tick_params(labelsize=25)

Method1, = plt.plot(epochs, acc1, linewidth =2.0, label='Method1') #3
Method2, = plt.plot(epochs, acc2, linewidth =2.0, label='Method2') #3
# LeNet, = plt.plot(epochs1, le_acc, linewidth =2.0, label='LeNet')
AlexNet, = plt.plot(epochs, alex_acc, linewidth =2.0, label='AlexNet') #2
GoogleNet, = plt.plot(epochs, google_acc, linewidth =2.0, label='GoogleNet') #3
VGG16, = plt.plot(epochs, vgg_acc, linewidth =2.0, label='VGG16') #4
ResNet50, = plt.plot(epochs, res_acc, linewidth =2.0, label='ResNet50') #5
plt.xlabel("The number of epochs", fontsize=30)
plt.ylabel("The accuracy of target orientation", fontsize=30)
plt.legend(loc="right", fontsize=25)

# fig, ax = plt.subplots(figsize=(6.4, 0.32))
# ax.legend(handles=[Method1, Method2, LeNet, AlexNet, GoogleNet, VGG16, ResNet50], mode='expand', ncol=7, borderaxespad=0)
# plt.legend(bbox_to_anchor=(0.96, 0), loc=3, borderaxespad=0, fontsize=25)
# ax.axis('off')
plt.show()

#
plt.figure(figsize=(10,8))
plt.tick_params(labelsize=25)
plt.plot(0, 0)
plt.plot(0, 0)
# plt.plot(epochs1, le_loss, label='LeNet') #2 3 4 5
plt.plot(epochs, alex_loss, linewidth =2.0,label='AlexNet')
plt.plot(epochs, google_loss, linewidth =2.0,label='GoogleNet')
plt.plot(epochs, vgg_loss, linewidth =2.0,label='VGG16')
plt.plot(epochs, res_loss, linewidth =2.0,label='ResNet50')
plt.xlabel("The number of epochs", fontsize=30)
plt.ylabel("The loss of target orientation", fontsize=30)
plt.legend(loc="upper right", fontsize=25)
plt.show()