import numpy as np
import copy
import pickle
import random
import warnings
from PIL import Image
from tqdm import tqdm
# from scipy.io import loadmat
import seaborn as sns
import matplotlib.pyplot as plt

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
        self.weights = np.zeros((self.sample_dim, self.embedding_size), dtype=bool)

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
        if distribution_random:
            weights = weights * np.mean(data, axis=0)[:, None]
        yindices = code_dims_indexs[None, :]
        xindices = weights.argsort(axis=0)[-self.num_projections:, :]
        self.weights[xindices, yindices] = True  # sparse projection vectors
        self.max_index_of_generated_weights += code_dims_length

    def PNtoKC(self, train_data, test_data, reset_num, center_data=False):
        if center_data:
            data_mean = np.mean(train_data, axis=1)
            self.train_data = (train_data - data_mean[:, None])
            data_mean = np.mean(test_data, axis=1)
            self.test_data = (test_data - data_mean[:, None])
        else:
            self.train_data = train_data
            self.test_data = test_data
        all_activations = (self.train_data @ self.weights)

        xindices_traindata = np.arange(train_data.shape[0])[:, None]
        # self.yindices = all_activations.argsort(axis=1)[:, -self.hash_length:]
        self.yindices_traindata = all_activations.argsort(axis=1)[:, -128:]
        self.kc_activity1_traindata = np.zeros_like(all_activations, dtype=bool)
        self.kc_activity1_traindata[xindices_traindata, self.yindices_traindata] = True  # choose topk activations

        self.kc_activity2_traindata = np.zeros_like(all_activations)
        self.kc_activity2_traindata[xindices_traindata, self.yindices_traindata] = all_activations[xindices_traindata, self.yindices_traindata]

        self.kc_activity3_traindata = np.zeros_like(all_activations)
        self.kc_activity3_traindata[xindices_traindata, self.yindices_traindata] = all_activations[xindices_traindata, self.yindices_traindata]
        print('kc_traindata')

        self.kc_on_off_traindata = np.ones_like(all_activations, dtype=bool)
        restrainnum_kc = np.count_nonzero(self.kc_activity3_traindata, axis=0)
        restrainnum_kc_id = np.where(restrainnum_kc > reset_num)[0]

        self.kc_activity3_traindata[:, restrainnum_kc_id] = 0
        self.kc_on_off_traindata[:, restrainnum_kc_id] = 0

        all_activations_restrain = np.zeros_like(all_activations)
        all_activations_restrain[self.kc_on_off_traindata] = all_activations[self.kc_on_off_traindata]

        yindices_traindata = all_activations_restrain.argsort(axis=1)[:, -128:]
        self.kc_activity3_traindata[xindices_traindata, yindices_traindata] = all_activations_restrain[
            xindices_traindata, yindices_traindata]
        self.kc_activity3_traindata = self.kc_activity3_traindata / np.sum(self.kc_activity3_traindata, axis=1,
                                                                           keepdims=1)
        self.kc_activity3_traindata = (self.kc_activity3_traindata - np.min(self.kc_activity3_traindata)) / (
                np.max(self.kc_activity3_traindata) - np.min(self.kc_activity3_traindata))
        print('kc_traindata_restrain')

        all_activations = (self.test_data @ self.weights)
        xindices_testdata = np.arange(test_data.shape[0])[:, None]
        self.yindices_testdata = all_activations.argsort(axis=1)[:, -128:]
        self.kc_activity1_testdata= np.zeros_like(all_activations, dtype=bool)
        self.kc_activity1_testdata[xindices_testdata, self.yindices_testdata] = True  # choose topk activations

        self.kc_activity2_testdata = np.zeros_like(all_activations)
        self.kc_activity2_testdata[xindices_testdata, self.yindices_testdata] = all_activations[xindices_testdata, self.yindices_testdata]

        self.kc_activity3_testdata = np.zeros_like(all_activations)
        self.kc_activity3_testdata[xindices_testdata, self.yindices_testdata] = all_activations[xindices_testdata, self.yindices_testdata]
        print('kc_testdata')

        self.kc_on_off_testdata = np.ones_like(all_activations, dtype=bool)
        restrainnum_kc = np.count_nonzero(self.kc_activity3_testdata, axis=0)
        restrainnum_kc_id = np.where(restrainnum_kc > reset_num)[0]

        self.kc_activity3_testdata[:, restrainnum_kc_id] = 0
        self.kc_on_off_testdata[:, restrainnum_kc_id] = 0

        all_activations_restrain = np.zeros_like(all_activations)
        all_activations_restrain[self.kc_on_off_testdata] = all_activations[self.kc_on_off_testdata]

        yindices_testdata = all_activations_restrain.argsort(axis=1)[:, -128:]
        self.kc_activity3_testdata[xindices_testdata, yindices_testdata] = all_activations_restrain[xindices_testdata, yindices_testdata]
        self.kc_activity3_testdata = self.kc_activity3_testdata / np.sum(self.kc_activity3_testdata, axis=1,keepdims=1)
        self.kc_activity3_testdata = (self.kc_activity3_testdata - np.min(self.kc_activity3_testdata)) / (
                np.max(self.kc_activity3_testdata) - np.min(self.kc_activity3_testdata))
        print('kc_testdata_restrain')

    def KCtoMBON(self, train_data, train_lables, center_data=False):
        if center_data:
            data_mean = np.mean(train_data, axis=1)
            self.train_data = (train_data - data_mean[:, None])
        else:
            self.train_data = train_data

        self.kc_mbon_weight1 = np.zeros((self.embedding_size, 72), dtype=bool)
        self.kc_mbon_weight2 = np.zeros((self.embedding_size, 72))
        self.kc_mbon_weight3 = np.zeros((self.embedding_size, 72))
        lr = np.ones((self.embedding_size, 72))


        for i in range(train_data.shape[0]):
            indices1 = np.argsort(self.kc_activity1_traindata[i])[-128:]  # 第i个图片对应的最强的2个KC的下标
            self.kc_mbon_weight1[indices1, train_lables[i]] = True  # 第i个图片对应的最强的2个KC连接到mbon上（第i个图片对应的mbon）

            indices2 = np.argsort(self.kc_activity2_traindata[i])[-128:]
            self.kc_mbon_weight2[indices2, train_lables[i]] = self.kc_mbon_weight2[indices2, train_lables[i]] + self.kc_activity2_traindata[
                i, indices2]


            indices3 = np.argsort(self.kc_activity3_traindata[i])[-128:]
            active_kc = self.kc_activity3_traindata[i, indices3]
            for j in range(i + 1):
                lr[indices3, train_lables[j]] = lr[indices3, train_lables[j]] * 0.9999  # 0.9999 0.95
            self.kc_mbon_weight3[indices3, train_lables[i]] = np.multiply(
                (active_kc - self.kc_mbon_weight3[indices3, train_lables[i]]), lr[indices3, train_lables[i]]) + self.kc_mbon_weight3[
                                                           indices3, train_lables[i]]


        # kc_activity1: max 01; kc_activity1: max 01
        self.img_mbon1_traindata = (self.kc_activity1_traindata @ self.kc_mbon_weight1)  # 离散 离散
        self.img_mbon2_traindata = np.matmul(self.kc_activity2_traindata, self.kc_mbon_weight1)  # 连续 离散
        self.img_mbon3_traindata = np.matmul(self.kc_activity2_traindata, self.kc_mbon_weight2)  # 连续 连续
        self.img_mbon4_traindata = np.matmul(self.kc_activity3_traindata, self.kc_mbon_weight3)  # 连续 连续 学习率
        print('mbon_traindata')

        self.img_mbon1_testdata = (self.kc_activity1_testdata @ self.kc_mbon_weight1)  # 离散 离散
        self.img_mbon2_testdata = np.matmul(self.kc_activity2_testdata, self.kc_mbon_weight1)  # 连续 离散
        self.img_mbon3_testdata = np.matmul(self.kc_activity2_testdata, self.kc_mbon_weight2)  # 连续 连续
        self.img_mbon4_testdata = np.matmul(self.kc_activity3_testdata, self.kc_mbon_weight3)  # 连续 连续 学习率
        print('mbon_testdata')

    def plot_maxMB(self, images, lables, max_5_mbons):
        plt.axes().get_xaxis().set_visible(False)  # 隐藏x坐标轴
        plt.axes().get_yaxis().set_visible(False)  # 隐藏y坐标轴
        lables = lables
        max_5_mbons = max_5_mbons

        for i in range(len(images)):
            plt.subplot(1, len(images), i + 1)
            plt.imshow(images[i].reshape((128, 128, 3)) / 255)
            xlabel = 'ori_angle' + str(lables[i] * 5)
            plt.xlabel(xlabel)
            title = 'max_5_mbons' + str(max_5_mbons[i] * 5)
            plt.title(title)
        plt.show()

    def predict(self, images, lables, images_nonave):
        right_angle1 = 0
        right_angle2 = 0
        means=[]
        vars=[]
        for img_id in range(images.shape[0]):
            max_indices1 = np.argsort(self.img_mbon4_testdata[img_id])[-2:]
            max_indices2 = np.argsort(self.img_mbon2_testdata[img_id])[-2:]
            if ((lables[img_id]-1) in max_indices1) or ((lables[img_id]+1) in max_indices1):  # 方法3
                right_angle1 = right_angle1 + 1
            if ((lables[img_id]-1) in max_indices2) or ((lables[img_id]+1) in max_indices2):  # 方法3
                right_angle2 = right_angle2 + 1
            # if (lables[img_id]) in max_indices1:  # 方法3
            #     right_angle1 = right_angle1 + 1
            # if (lables[img_id]) in max_indices2:  # 方法3
            #     right_angle2 = right_angle2 + 1
        acc1 = right_angle1 / images.shape[0]
        acc2 = right_angle2 / images.shape[0]
        print('predict')
        print(acc1)
        print(acc2)
        print('end')
        return acc

    def lifelong_predict(self, i, images, lables, images_nonave):
        acc1 = np.zeros(i)
        acc2 = np.zeros(i)
        acc3 = np.zeros(i)
        for id in range(i):
            right_angle2 = 0
            right_angle3 = 0
            right_angle4 = 0
            start_index = id * 72
            end_index = min((id + 1) * 72, images.shape[0])
            for img_id in range(start_index, end_index):
                max_indices2 = np.argsort(self.img_mbon2_traindata[img_id])[-1:]

                max_indices3 = np.argsort(self.img_mbon3_traindata[img_id])[-1:]

                max_indices4 = np.argsort(self.img_mbon4_traindata[img_id])[-1:]

                if lables[img_id] in max_indices2:
                    right_angle2 = right_angle2 + 1

                if lables[img_id] in max_indices3:
                    right_angle3 = right_angle3 + 1

                if lables[img_id] in max_indices4:
                    right_angle4 = right_angle4 + 1

            acc1[id] = right_angle4 / 72  # KC浮点数-Syn浮点数（权重）-MBON浮点数

            acc2[id] = right_angle3 / 72  # KC浮点数-Syn浮点数（直接相加）-MBON浮点数

            acc3[id] = right_angle2 / 72  # KC浮点数-Syn二进制-MBON浮点数
        return acc1, acc2, acc3

def single_test(hash_length, embedding_size, train_set, test_set, sampling_ratio,
                all_expriments, train_angles, test_angles, images_nonave, if_center_data=True):
    seed = hash_length * embedding_size * sampling_ratio
    random.seed(seed)
    np.random.seed(int(seed))
    model = {}
    functions = {'Fly': FlyLSH}
    sample_dim = train_set.shape[1]
    reset_num = int(0.25*train_set.shape[0])
    for expriment in all_expriments:  # ['Fly', 'FlylshDevelop', 'FlylshDevelopThreshold','FlylshDevelopThresholdRandomChoice']
        model[expriment] = functions[expriment](sample_dim, hash_length, sampling_ratio, embedding_size)
        model[expriment].generate_weights(train_set)
        model[expriment].PNtoKC(train_set, test_set, reset_num=int(reset_num/10), center_data=if_center_data)
        model[expriment].KCtoMBON(train_set, train_angles, center_data=if_center_data)
        # i=0
        # start = 0
        # end = test_set.shape[0]
        # model[expriment].MBONtoCANN(start, end, images_nonave)
        acc = model[expriment].predict(test_set, test_angles, images_nonave)
        # acc = model[expriment].predict(train_set, train_angles, images_nonave)
        print(i)
        return acc


def lifelong_test(hash_length, embedding_size, training_data, testing_data, sampling_ratio,
                  all_expriments, lables, images_nonave, if_center_data=True):
    seed = hash_length * embedding_size * sampling_ratio
    random.seed(seed)
    np.random.seed(int(seed))
    model = {}
    functions = {'Fly': FlyLSH}
    sample_dim = training_data.shape[1]
    for expriment in all_expriments:  # ['Fly', 'FlylshDevelop', 'FlylshDevelopThreshold','FlylshDevelopThresholdRandomChoice']

        accs1 = []
        accs2 = []
        accs3 = []
        for i in range(0, 87):  # 13 #87
            # acc = np.zeros((i + 1, 3))
            images = training_data[:72 * (i + 1), :]
            lables_ = lables[:72 * (i + 1)]
            model[expriment] = functions[expriment](sample_dim, hash_length, sampling_ratio, embedding_size)
            model[expriment].generate_weights(images)
            model[expriment].PNtoKC(images, center_data=if_center_data)
            model[expriment].KCtoMBON(images, lables, center_data=if_center_data)
            acc1, acc2, acc3 = model[expriment].lifelong_predict(i + 1, images, lables_, images_nonave)
            accs1.append(acc1)
            accs2.append(acc2)
            accs3.append(acc3)
            print(i)

            # for j in range(i+1):
            #     images_j = images[72 * j:72 * (j + 1), :]
            #     lables_j = lables_[72 * j:72 * (j + 1)]
            # for a in range(72):
            #     plt.imshow(images_j[a].reshape((128, 128, 3)) / 255)
            #     plt.show()

            #     acc[j] = model[expriment].lifelong_predict(j, images_j, lables_j, images_nonave)
            # accs.append(acc)
        result1 = np.zeros((87, 87))  # ((100, 100))#((87, 87))#((13, 13))
        for i, row in enumerate(accs1):
            result1[i][:len(row)] = row
        result2 = np.zeros((87, 87))
        for i, row in enumerate(accs2):
            result2[i][:len(row)] = row
        result3 = np.zero((87, 87))
        for i, row in enumerate(accs3):
            result3[i][:len(row)] = row
        np.savetxt('./saved_csv/coil_del/longlife_sim_me1.csv', result1, delimiter=',')
        np.savetxt('./saved_csv/coil_del/longlife_sim_me2.csv', result2, delimiter=',')
        np.savetxt('./saved_csv/coil_del/longlife_sim_me3.csv', result3, delimiter=',')

        # with open("./saved_csv/lifelong_accs_ave.txt", "w") as file:
        #     for item in accs:
        #         file.write(str(item) + "\n")
    # np.savetxt('longlife_accs.txt', accs, delimiter=',')
        return result3


def hugeimage_test(hash_length, embedding_size, training_data, testing_data, sampling_ratio,
                   all_expriments, lables, images_nonave, if_center_data=True):
    seed = hash_length * embedding_size * sampling_ratio
    random.seed(seed)
    np.random.seed(int(seed))
    model = {}
    functions = {'Fly': FlyLSH}
    sample_dim = training_data.shape[1]
    for expriment in all_expriments:  # ['Fly', 'FlylshDevelop', 'FlylshDevelopThreshold','FlylshDevelopThresholdRandomChoice']
        accnums = np.zeros(3)
        for i in range(0, 72001, 7200):
            images = training_data[7200 * i:7200 * (i + 1), :]
            lables_ = lables[7200 * i: 7200 * (i + 1)]
            model[expriment] = functions[expriment](sample_dim, hash_length, sampling_ratio, embedding_size)
            model[expriment].generate_weights(images)
            model[expriment].PNtoKC(images, center_data=if_center_data)
            model[expriment].KCtoMBON(images, lables_, center_data=if_center_data)
            accnum = model[expriment].predict_accnums(testing_data, lables, images_nonave)
            accnums = accnums + accnum
            print(accnum)
        acc = accnums / images.shape[0]
        return acc


if __name__ == '__main__':

    max_index = 10000
    sampling_ratio = 0.10
    nnn = 200  # number of nearest neighbours to compare, 2% of max_index as in paper
    hash_lengths = [512]
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

    train_set = []
    test_set = []
    train_angles = []
    test_angles = []
    images_sim_all = np.loadtxt('./saved_csv/coil_image_del.csv', delimiter=',')  # , max_rows=144 , max_rows=936 , max_rows=14800
    angles = np.loadtxt('./saved_csv/coil_lable_del.csv', dtype=int, delimiter=',')  # , max_rows=144
    lables_all = angles // 5  # sim
    for i in range(0, len(images_sim_all)):
        if i % 2 == 0:
            train_set.append(images_sim_all[i])
            train_angles.append(lables_all[i])
        else:
            test_set.append(images_sim_all[i])
            test_angles.append(lables_all[i])
    train_set = np.array(train_set)
    test_set = np.array(test_set)
    train_angles = np.array(train_angles)
    test_angles = np.array(test_angles)
    # lables_all = angles #coil-ori aloi
    print('loaded')
    images_nonave = images_sim_all
    images_ave = (images_sim_all - np.mean(images_sim_all, axis=1)[:, None])
    # 5.初始化准确率列表
    all_expriments = ['Fly']

    img_accs = []  # 87

    if lifelong:
        for hash_length in hash_lengths:  # k
            print('life-long')
            embedding_size = int(
                ratio * hash_length)  # int(10*input_dim) #20k or 10d  #20*[2, 4, 8, 12, 16, 20, 24, 28, 32]
            acc = lifelong_test(hash_length, embedding_size, train_set, test_set, sampling_ratio,
                                all_expriments, train_angles, test_angles, images_nonave, if_center_data)
            print(acc[:, :10])
            img_accs.append(acc)

    else:
        for hash_length in hash_lengths:
            print(hash_length)  # k
            embedding_size = int(20 * hash_length)  # int(10*input_dim) #20k or 10d
            acc_nonave = single_test(hash_length, embedding_size, train_set, test_set, sampling_ratio,
                                all_expriments, train_angles, test_angles, images_nonave, if_center_data)

            print(acc_nonave)
            np.savetxt('./saved_csv/coil_del_cann_acc_nonave.txt', acc_nonave, delimiter=',')

