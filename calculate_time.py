import numpy as np
import pickle
import random
import warnings
from PIL import Image
from tqdm import tqdm
# from scipy.io import loadmat
import seaborn as sns
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn



plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class FlyLSH(object):
    def __init__(self, sample_dim, hash_length, sampling_ratio, embedding_size):
        """
        data: uxd matrix
        hash_length: scalar
        sampling_ratio: fraction of input dims to sample from when producing a hash
        embedding_size: dimensionality of projection space, m
        Note that in Flylsh, the hash length and embedding_size are NOT the same
        whereas in usual LSH they are
        """
        self.sample_dim = sample_dim
        self.hash_length = hash_length
        self.embedding_size = embedding_size
        self.K = embedding_size // hash_length
        self.num_projections = 400  # int(sampling_ratio * self.sample_dim) #40

        self.maxl1distance = 2 * self.hash_length
        self.max_index_of_generated_weights = 0
        self.weights = torch.zeros((self.sample_dim, self.embedding_size), dtype=torch.double, device='cuda')
        self.kc_mbon_weight3 = torch.zeros((self.embedding_size, 72), dtype=torch.double, device='cuda')
        self.lr = torch.ones((self.embedding_size, 72), device='cuda')

    def generate_weights(self, data, code_dims=None, distribution_random=False):
        if code_dims is None:
            code_dims_indexs = np.arange(self.max_index_of_generated_weights, self.embedding_size, dtype=np.int64)
            code_dims_length = int(self.embedding_size - self.max_index_of_generated_weights)
        elif np.isscalar(code_dims):
            if code_dims < 1:
                code_dims = int(code_dims * self.embedding_size)
            if self.max_index_of_generated_weights >= code_dims:
                warnings.warn('Unexpected modification on existing connection!, max_index=%d, code_dims=%d' % (
                    self.max_index_of_generated_weights, code_dims))
            code_dims_indexs = np.arange(self.max_index_of_generated_weights, code_dims, dtype=np.long)
            code_dims_length = int(code_dims - self.max_index_of_generated_weights)
        else:
            if not np.all(np.less(self.max_index_of_generated_weights, code_dims)):
                warnings.warn('Unexpected modification on existing connection!, max_index=%d, code_dims=%d' % (
                    self.max_index_of_generated_weights, code_dims))
            code_dims_indexs = code_dims
            code_dims_length = len(code_dims)

        weights = np.random.random((self.sample_dim, code_dims_length))

        yindices = code_dims_indexs[None, :]
        xindices = weights.argsort(axis=0)[-self.num_projections:, :]
        self.weights[xindices, yindices] = 1  # sparse projection vectors
        self.max_index_of_generated_weights += code_dims_length

    def PNtoKC(self, data, center_data=False):
        self.data = data
        all_activations = torch.matmul(self.data, self.weights)
        xindices = torch.arange(data.shape[0])[:, None]
        _, yindices = torch.sort(all_activations, dim=1)
        self.yindices = yindices[:, -128:]
        # self.kc_activity1 = np.zeros_like(all_activations, dtype=bool)
        # self.kc_activity1[xindices, self.yindices] = True  # choose topk activations
        #
        # self.kc_activity2 = np.zeros_like(all_activations)
        # self.kc_activity2[xindices, self.yindices] = all_activations[xindices, self.yindices]

        self.kc_activity3 = torch.zeros_like(all_activations)
        self.kc_activity3[xindices, self.yindices] = all_activations[xindices, self.yindices]
        self.kc_activity3 = self.kc_activity3 / torch.sum(self.kc_activity3, axis=1, keepdims=True)
        self.kc_activity3 = (self.kc_activity3 - torch.min(self.kc_activity3)) / (
                    torch.max(self.kc_activity3) - torch.min(self.kc_activity3))
        print('kc')

    def KCtoMBON(self, data, lables, center_data=False):

        #
        # self.kc_mbon_weight1 = np.zeros((self.embedding_size, 72), dtype=bool)
        # self.kc_mbon_weight2 = np.zeros((self.embedding_size, 72))



        #
        #
        # kc_weight_mbon = self.kc_activity2 * weight

        for i in range(data.shape[0]):
            print(i)
            indices = self.yindices[i]  # 第i个图片对应的最强的2个KC的下标
            # self.kc_mbon_weight1[indices, lables[i]] = True  # 第i个图片对应的最强的2个KC连接到mbon上（第i个图片对应的mbon）
            # self.kc_mbon_weight2[indices, lables[i]] = self.kc_mbon_weight2[indices, lables[i]] + self.kc_activity2[
            #     i, indices]
            active_kc = self.kc_activity3[i, self.yindices[i]]
            for j in range(i + 1):
                self.lr[indices, lables[j]] = self.lr[indices, lables[j]] * 0.9999  # 0.9999 0.95
            self.kc_mbon_weight3[indices, lables[i]] = torch.multiply(
                (active_kc - self.kc_mbon_weight3[indices, lables[i]]), self.lr[indices, lables[i]]) + self.kc_mbon_weight3[
                                                           indices, lables[i]]

        # kc_activity1: max 01; kc_activity1: max 01
        # self.img_mbon2 = np.matmul(self.kc_activity2, self.kc_mbon_weight1)  # 连续 离散
        self.img_mbon4 = torch.matmul(self.kc_activity3, self.kc_mbon_weight3)  # 连续 连续 学习率
        print('mbon')


    def predict(self, images, lables, images_nonave):
        right_angle2 = 0
        right_angle3 = 0
        right_angle4 = 0
        for img_id in range(images.shape[0]):
            # # 方法3
            # max_indices2 = np.argsort(self.img_mbon2[img_id])[-1:]


            # 方法1
            max_indices4 = torch.argsort(self.img_mbon4[img_id], descending=True)[:1].cuda()

            #
            # if lables[img_id] in max_indices2:  # 方法3
            #     right_angle2 = right_angle2 + 1


            if lables[img_id] in max_indices4:  # 方法1
                right_angle4 = right_angle4 + 1
            # print('end')

        acc = np.zeros(3)

        acc[0] = right_angle4 / images.shape[0]  # KC浮点数-Syn浮点数（权重）-MBON浮点数 方法1

        # acc[2] = right_angle2 / images.shape[0]  # KC浮点数-Syn二进制-MBON浮点数  方法3
        return acc



def single_test(hash_length, embedding_size, training_data, testing_data, sampling_ratio,
                all_expriments, lables, images_nonave, if_center_data=True):
    seed = hash_length * embedding_size * sampling_ratio
    random.seed(seed)
    np.random.seed(int(seed))
    model = {}
    functions = { 'Fly': FlyLSH}
    sample_dim = training_data.shape[1]
    for expriment in all_expriments:  # ['Fly', 'FlylshDevelop', 'FlylshDevelopThreshold','FlylshDevelopThresholdRandomChoice']
        # print(expriment)
        if expriment == 'LSH':
            print('no')

        else:
            # t0 = time.time()
            model[expriment] = functions[expriment](sample_dim, hash_length, sampling_ratio, embedding_size)
            t0 = time.time()
            model[expriment].generate_weights(training_data)
            model[expriment].PNtoKC(testing_data, center_data=if_center_data)
            model[expriment].KCtoMBON(testing_data, lables, center_data=if_center_data)

            acc = model[expriment].predict(testing_data, lables, images_nonave)
            t1 = time.time()
            training_time = t1 - t0
            print(training_time)
        return acc


if __name__ == '__main__':
    torch.cuda.current_device()  # 当前使用gpu的设备号
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_index = 10000
    sampling_ratio = 0.10
    nnn = 200  # number of nearest neighbours to compare, 2% of max_index as in paper
    hash_lengths = [512]  # 2, 32, 64, 128, 256,
    number_of_tests = 1
    ratio = 20
    result_root_path = "./Results/NoPreprocessing/"  # "./Results/main/"
    lifelong = False
    hugeimge = False
    sorted_data = False
    number_of_process = 1

    threshold = 10
    if_center_data = False
    all_MAPs = {}

    images_sim_all = np.loadtxt('./saved_csv/coil_image_del.csv',
                                delimiter=',')  # , max_rows=144 , max_rows=936 , max_rows=14800
    angles = torch.from_numpy(np.loadtxt('./saved_csv/coil_lable_del.csv', dtype=int, delimiter=',')).cuda()  # , max_rows=144
    lables_all = angles // 5  # sim

    # lables_all = angles #coil-ori aloi
    print('loaded')
    images_nonave = torch.from_numpy(images_sim_all).cuda()
    # 5.初始化准确率列表
    all_expriments = ['Fly']


    for hash_length in hash_lengths:
        print(hash_length)  # k

        embedding_size = int(20 * hash_length)  # int(10*input_dim) #20k or 10d
        acc_nonave = single_test(hash_length, embedding_size, images_nonave, images_nonave, sampling_ratio,
                                 all_expriments, lables_all, images_nonave, if_center_data)


