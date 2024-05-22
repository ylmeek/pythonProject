import numpy as np
import pickle
import random
import warnings
from PIL import Image
from tqdm import tqdm
#from scipy.io import loadmat
import seaborn as sns
import matplotlib.pyplot as plt
import  time

plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

class LSH(object):
    def __init__(self, sample_dim, hash_length):
        """
        data: uxd matrix
        hash_length: scalar
        sampling_ratio: fraction of input dims to sample from when producing a hash
        (ratio of PNs that each KC samples from)
        embedding_size: dimensionality of projection space, m
        """
        self.sample_dim = sample_dim
        self.hash_length = hash_length
        self.maxl1distance = 2 * self.hash_length
        self.weights = np.zeros((self.sample_dim, self.hash_length))
        self.max_index_of_generated_weights = 0

    def generate_weights(self, code_dims=None):
        if code_dims is None:
            # Develop til full
            self.weights[:, self.max_index_of_generated_weights:] = \
                np.random.random((self.sample_dim, self.hash_length - self.max_index_of_generated_weights))
            self.max_index_of_generated_weights = self.hash_length
        elif np.isscalar(code_dims):
            if code_dims < 1:
                code_dims = int(code_dims * self.hash_length)
            # Develop code_dims more
            if self.max_index_of_generated_weights >= code_dims:
                warnings.warn('Unexpected modification on existing connection!')
            self.weights[:, self.max_index_of_generated_weights:self.max_index_of_generated_weights + code_dims] = \
                np.random.random((self.sample_dim, code_dims))
            self.max_index_of_generated_weights += code_dims

        else:
            # Develop specific code dims following the elements in code_dims as indexes
            if not np.all(np.less(self.max_index_of_generated_weights, code_dims)):
                warnings.warn('Unexpected modification on existing connection!')
            self.weights[:, code_dims] = np.random.random((self.sample_dim, len(code_dims)))
            self.max_index_of_generated_weights += len(code_dims)

    def hashing(self, data):
        self.data = data - np.mean(data, axis=1)[:, None]
        self.hashes = (data @ self.weights) > 0

    def query(self, qidx, nnn, not_olap=False):
        L1_distances = np.sum(np.abs(self.hashes[qidx, :] ^ self.hashes), axis=1)
        nnn = min(self.hashes.shape[0], nnn)
        if not_olap:
            no_overlaps = np.sum(L1_distances == self.maxl1distance)
            return no_overlaps

        NNs = L1_distances.argsort()
        NNs = NNs[(NNs != qidx)][:nnn]
        # print(L1_distances[NNs]) #an interesting property of this hash is that the L1 distances are always even
        return NNs

    def true_nns(self, qidx, nnn):
        sample = self.data[qidx, :]
        tnns = np.sum((self.data - sample) ** 2, axis=1).argsort()[:nnn + 1]
        tnns = tnns[(tnns != qidx)]
        if nnn < self.data.shape[0]:
            assert len(tnns) == nnn, 'nnn={}'.format(nnn)
        return tnns

    def construct_true_nns(self, indices, nnn):
        all_NNs = np.zeros((len(indices), nnn))
        for idx1, idx2 in enumerate(indices):
            all_NNs[idx1, :] = self.true_nns(idx2, nnn)
        return all_NNs

    def AP(self, predictions, truth):
        assert len(predictions) == len(truth) or len(predictions) == self.hashes.shape[0]
        # removed conversion to list in next line:
        precisions = [len((set(predictions[:idx]).intersection(set(truth[:idx])))) / idx for \
                      idx in range(1, len(truth) + 1)]
        return np.mean(precisions)

    def PR(self, qidx, truth, atindices):
        """truth should be a set"""
        L1_distances = np.sum((self.hashes[qidx, :] ^ self.hashes), axis=1)
        NNs = L1_distances.argsort()
        NNs = NNs[(NNs != qidx)]
        # predictions=NNs
        recalls = np.arange(1, len(truth) + 1)
        all_recalls = [len(set(NNs[:idx]) & truth) for idx in atindices]
        # all_recalls.append(len(set(NNs)&truth))
        # all_recalls=[len(set(predictions[:idx])&truth) for idx in range(1,self.hashes.shape[0]+1)]
        # indices=[all_recalls.index(recall) for recall in recalls]
        precisions = [recall / (idx + 1) for idx, recall in zip(atindices, all_recalls)]
        # this_pr=odict({l:(p,r) for l,p,r in zip(atL1,precisions,recalls)})
        return precisions, all_recalls  # (precisions,all_recalls)

    def ROC(self, qidx, truth, atindices):
        """x: False positive rate, y: True positive rate, truth should be a set"""
        L1_distances = np.sum((self.hashes[qidx, :] ^ self.hashes), axis=1)
        NNs = L1_distances.argsort()
        NNs = NNs[(NNs != qidx)]
        x, y = [], []
        for idx in atindices:
            ntruepos = len((set(NNs[:idx]) & truth))  # number of positives correctly classified
            nfalseneg = idx - ntruepos  # number of negatives incorrectly classified
            tpr = ntruepos / len(truth)  # positives correctly classified / total positives
            fpr = nfalseneg / (len(NNs) - len(truth))  # negatives incorrectly classified / total negatives
            x.append(fpr)
            y.append(tpr)
        return x, y

    def findmAP_given_true(self, nnn, n_points, all_NNs):
        return np.mean(self.findAPs_given_true(nnn, n_points, all_NNs))

    def findAPs_given_true(self, nnn, n_points, all_NNs):
        sample_indices = np.random.choice(self.data.shape[0], n_points)
        self.allAPs = []
        for eidx, didx in enumerate(sample_indices):
            # eidx: enumeration id, didx: index of sample in self.data
            this_nns = self.query(didx, nnn)
            # print(len(this_nns))
            this_AP = self.AP(list(this_nns), list(all_NNs[didx, :]))
            # print(this_AP)
            self.allAPs.append(this_AP)
        return self.allAPs

    def findmAP_given_true_labels(self, nnn, n_points, all_NNs, labels, label_set=None):
        if label_set is None:
            label_set = set(labels)
        self.findAPs_given_true(nnn, n_points, all_NNs)
        self.mAP_per_lable = {}
        for a_label in label_set:
            self.mAP_per_lable[a_label] = np.mean(np.array(self.allAPs)[labels == a_label])
        return self.mAP_per_lable

    def findmAP(self, nnn, n_points):
        start = np.random.randint(low=0, high=self.data.shape[0] - n_points)
        sample_indices = np.random.choice(self.data.shape[0], n_points)
        all_NNs = self.construct_true_nns(sample_indices, nnn)
        self.allAPs = []
        for eidx, didx in enumerate(sample_indices):
            # eidx: enumeration id, didx: index of sample in self.data
            this_nns = self.query(didx, nnn)
            # print(len(this_nns))
            this_AP = self.AP(list(this_nns), list(all_NNs[eidx, :]))
            # print(this_AP)
            self.allAPs.append(this_AP)
        return np.mean(self.allAPs)

    def findZKk(self, n_points):
        """
        ZKk is the number of vectors whose overlap with a specific vector is zero
        """
        sample_indices = np.random.choice(self.data.shape[0], n_points)
        no_overlaps = []
        for eidx, didx in enumerate(sample_indices):
            no_overlaps.append(self.query(didx, -20, not_olap=True))
        return np.mean(no_overlaps)

    def computePRC(self, n_points=1, nnn=200, atindices=None):
        """
        This function calculates precision-recall metrics for model
        """

        def replacenans(x):
            nanidxs = [idx for idx in range(len(x)) if np.isnan(x[idx])]
            notnang = lambda idx: [nidx for nidx in range(idx + 1, len(x)) if nidx not in nanidxs][0]
            notnans = lambda idx: [nidx for nidx in range(idx) if nidx not in nanidxs][-1]
            if len(nanidxs) == 0:
                return x
            else:
                for nanidx in nanidxs:
                    if nanidx == 0:
                        x[nanidx] = x[notnang(nanidx)]
                    else:
                        x[nanidx] = (x[notnang(nanidx)] + x[notnans(nanidx)]) / 2
                return x

        sample_indices = np.random.choice(self.data.shape[0], n_points)
        all_NNs = self.construct_true_nns(sample_indices, nnn)
        self.allprecisions = np.zeros((n_points, len(atindices)))
        self.allrecalls = np.zeros((n_points, len(atindices)))
        # allprs=odict({l:[[],[]] for l in atL1})
        for eidx, didx in enumerate(sample_indices):
            """eidx: enumeration id, didx: index of sample in self.data"""
            this_p, this_r = self.PR(didx, set(all_NNs[eidx, :]), atindices)
            self.allprecisions[eidx, :] = this_p
            self.allrecalls[eidx, :] = this_r

        return [self.allprecisions.mean(axis=0),
                self.allrecalls.mean(axis=0)]  # replacenans([np.nanmean(v) for _,v in allprcs.items()])

    def computeROC(self, n_points=1, nnn=200, atindices=None):
        """
        This function calculates receiver operator characteristics (ROC)
        """
        sample_indices = np.random.choice(self.hashes.shape[0], n_points)
        all_NNs = self.construct_true_nns(sample_indices, nnn)
        alltprs = np.zeros((n_points, len(atindices)))
        allfprs = np.zeros((n_points, len(atindices)))
        for eidx, didx in enumerate(sample_indices):
            this_fpr, this_tpr = self.ROC(didx, set(all_NNs[eidx, :]), atindices)
            allfprs[eidx, :] = this_fpr
            alltprs[eidx, :] = this_tpr
        return [allfprs.mean(axis=0), alltprs.mean(axis=0)]

class FlyLSH(LSH):
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
        self.num_projections = 400#int(sampling_ratio * self.sample_dim) #40

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

    def PNtoKC(self, data, center_data=False):
        if center_data:
            data_mean = np.mean(data, axis=1)
            self.data = (data - data_mean[:, None])
        else:
            self.data = data
        all_activations = (self.data @ self.weights)
        xindices = np.arange(data.shape[0])[:, None]
        #self.yindices = all_activations.argsort(axis=1)[:, -self.hash_length:]
        self.yindices = all_activations.argsort(axis=1)[:, -128:]
        self.kc_activity1 = np.zeros_like(all_activations, dtype=bool)
        self.kc_activity1[xindices, self.yindices] = True  # choose topk activations

        self.kc_activity2 = np.zeros_like(all_activations)
        self.kc_activity2[xindices, self.yindices] = all_activations[xindices, self.yindices]

        self.kc_activity3 = np.zeros_like(all_activations)
        self.kc_activity3[xindices, self.yindices] = all_activations[xindices, self.yindices]
        self.kc_activity3 = self.kc_activity3 / np.sum(self.kc_activity3, axis=1, keepdims=1)
        self.kc_activity3 = (self.kc_activity3 - np.min(self.kc_activity3))/(np.max(self.kc_activity3)-np.min(self.kc_activity3))
        print('kc')


    def KCtoMBON(self, data, lables, center_data=False):
        if center_data:
            data_mean = np.mean(data, axis=1)
            self.data = (data - data_mean[:, None])
        else:
            self.data = data

        self.kc_mbon_weight1 = np.zeros((self.embedding_size, 72), dtype=bool)
        self.kc_mbon_weight2 = np.zeros((self.embedding_size, 72))
        self.kc_mbon_weight3 = np.zeros((self.embedding_size, 72))
        lr = np.ones((self.embedding_size, 72))

        #lr = np.ones_like(self.kc_activity3)
        # weight = np.zeros_like(self.kc_activity3)
        # for j in range(self.data.shape[0]):
        #     # active_kc = self.kc_activity2[j, self.yindices[j]]
        #     # weight[j, self.yindices[j]] = (active_kc - weight[j, self.yindices[j]]) * lr + weight[j, self.yindices[j]]
        #     # weight = (self.hashes2-weight) * lr + weight
        #     active_kc = self.kc_activity3[j, self.yindices[j]]
        #     weight[j, self.yindices[j]] = np.multiply((active_kc - weight[j, self.yindices[j]]),  lr[j, self.yindices[j]]) + weight[j, self.yindices[j]]
        #     for i in range(j+1):
        #         lr[i, self.yindices[i]] = lr[i, self.yindices[i]] * 0.99
        #
        #
        # kc_weight_mbon = self.kc_activity2 * weight

        for i in range(data.shape[0]):
            indices = self.yindices[i] #第i个图片对应的最强的2个KC的下标
            self.kc_mbon_weight1[indices, lables[i]] = True#第i个图片对应的最强的2个KC连接到mbon上（第i个图片对应的mbon）
            self.kc_mbon_weight2[indices, lables[i]] = self.kc_mbon_weight2[indices, lables[i]] + self.kc_activity2[i, indices]
            active_kc = self.kc_activity3[i, self.yindices[i]]
            for j in range(i + 1):
                lr[indices, lables[j]] = lr[indices, lables[j]] * 0.9999#0.9999 0.95
            self.kc_mbon_weight3[indices, lables[i]] = np.multiply((active_kc-self.kc_mbon_weight3[indices, lables[i]] ), lr[indices, lables[i]]) + self.kc_mbon_weight3[indices, lables[i]]
            #self.kc_mbon_weight3[indices, lables[i]] = self.kc_mbon_weight3[indices, lables[i]] + kc_weight_mbon[i, indices]


        #kc_activity1: max 01; kc_activity1: max 01
        self.img_mbon1 = (self.kc_activity1 @ self.kc_mbon_weight1) #离散 离散
        self.img_mbon2 = np.matmul(self.kc_activity2, self.kc_mbon_weight1) #连续 离散
        self.img_mbon3 = np.matmul(self.kc_activity2, self.kc_mbon_weight2) #连续 连续
        self.img_mbon4 = np.matmul(self.kc_activity3, self.kc_mbon_weight3) #连续 连续 学习率
        print('mbon')


    def plot_maxMB(self, images, lables, max_5_mbons):
        plt.axes().get_xaxis().set_visible(False)  # 隐藏x坐标轴
        plt.axes().get_yaxis().set_visible(False)  # 隐藏y坐标轴
        lables = lables
        max_5_mbons =max_5_mbons

        for i in range(len(images)):
            plt.subplot(1, len(images), i+1)
            plt.imshow(images[i].reshape((128, 128, 3))/255)
            xlabel = 'ori_angle'+str(lables[i]*5)
            plt.xlabel(xlabel)
            title = 'max_5_mbons'+str(max_5_mbons[i]*5)
            plt.title(title)
        plt.show()

    def predict(self, images, lables, images_nonave):
        right_angle2 = 0
        right_angle3 = 0
        right_angle4 = 0
        max_5_mbons = []
        for img_id in range(images.shape[0]):

            max_5_mbon = np.zeros((2,5), dtype=int)

            #方法3
            max_indices2 = np.argsort(self.img_mbon2[img_id])[-1:]
            max_5_mbon[1] = np.argsort(self.img_mbon2[img_id])[-5:][::-1] #升序变降序

            #方法2
            max_indices3 = np.argsort(self.img_mbon3[img_id])[-1:]
            # max_5_mbon[1] = np.argsort(self.img_mbon3[img_id])[-5:][::-1]

            #方法1
            max_indices4 = np.argsort(self.img_mbon4[img_id])[-1:]
            max_5_mbon[0] = np.argsort(self.img_mbon4[img_id])[-5:][::-1]


            if img_id % 5 == 0:
                self.plot_maxMB(images_nonave[img_id-5:img_id], lables[img_id-5:img_id], max_5_mbons[img_id-5:img_id])
            max_5_mbons.append(max_5_mbon)


            if lables[img_id] in max_indices2:#方法3
                right_angle2 = right_angle2 + 1

            if lables[img_id] in max_indices3:#方法2
                right_angle3 = right_angle3 + 1
            
            if lables[img_id] in max_indices4:#方法1
                right_angle4 = right_angle4 + 1
            # print('end')

        # acc = np.zeros(6)
        # acc[0] = right_angle1 / images.shape[0] #KC二进制-Syn二进制-MBON二进制
        #
        # acc[1] = right_angle2 / images.shape[0] #KC浮点数-Syn二进制-MBON浮点数
        # acc[2] = right_angle3 / images.shape[0] #KC浮点数-Syn浮点数（直接相加）-MBON浮点数
        # acc[3] = right_angle4 / images.shape[0] #KC浮点数-Syn浮点数（权重）-MBON浮点数
        #
        # acc[4] = right_angle5 / images.shape[0] #KC二进制-Syn二进制-MBON二进制（最小）
        # acc[5] = right_angle6 / images.shape[0] #KC浮点数-Syn二进制-MBON浮点数（最小）
        acc = np.zeros(3)

        acc[0] = right_angle4 / images.shape[0] #KC浮点数-Syn浮点数（权重）-MBON浮点数 方法1

        acc[1] = right_angle3 / images.shape[0] #KC浮点数-Syn浮点数（直接相加）-MBON浮点数 方法2

        acc[2] = right_angle2 / images.shape[0] #KC浮点数-Syn二进制-MBON浮点数  方法3
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
                max_indices2 = np.argsort(self.img_mbon2[img_id])[-1:]

                max_indices3 = np.argsort(self.img_mbon3[img_id])[-1:]

                max_indices4 = np.argsort(self.img_mbon4[img_id])[-1:]

                if lables[img_id] in max_indices2:
                    right_angle2 = right_angle2 + 1

                if lables[img_id] in max_indices3:
                    right_angle3 = right_angle3 + 1

                if lables[img_id] in max_indices4:
                    right_angle4 = right_angle4 + 1

            acc1[id] = right_angle4 / 72 #KC浮点数-Syn浮点数（权重）-MBON浮点数

            acc2[id] = right_angle3 / 72 #KC浮点数-Syn浮点数（直接相加）-MBON浮点数

            acc3[id] = right_angle2 / 72 #KC浮点数-Syn二进制-MBON浮点数
        return acc1, acc2, acc3

    def predict_accnums(self, images, lables, images_nonave):
        right_angle2 = 0
        right_angle3 = 0
        right_angle4 = 0
        max_5_mbons = []

        for img_id in range(images.shape[0]):
            max_5_mbon = np.zeros((3, 5), dtype=int)
            a = self.img_mbon2[img_id]
            # 方法3
            max_indices2 = np.argsort(self.img_mbon2[img_id])[-1:]
            max_5_mbon[0] = np.argsort(self.img_mbon2[img_id])[-5:][::-1]  # 升序变降序

            # 方法2
            max_indices3 = np.argsort(self.img_mbon3[img_id])[-1:]
            max_5_mbon[1] = np.argsort(self.img_mbon3[img_id])[-5:][::-1]

            # 方法1
            max_indices4 = np.argsort(self.img_mbon4[img_id])[-1:]
            max_5_mbon[2] = np.argsort(self.img_mbon4[img_id])[-5:][::-1]

            if lables[img_id] in max_indices2:  # 方法3
                right_angle2 = right_angle2 + 1

            if lables[img_id] in max_indices3:  # 方法2
                right_angle3 = right_angle3 + 1

            if lables[img_id] in max_indices4:  # 方法1
                right_angle4 = right_angle4 + 1

        acc = np.zeros(3)

        acc[0] = right_angle4

        acc[1] = right_angle3

        acc[2] = right_angle2
        return acc

def single_test(hash_length, embedding_size, training_data, testing_data, sampling_ratio,
                all_expriments, lables, images_nonave, if_center_data=True):
    seed = hash_length * embedding_size * sampling_ratio
    random.seed(seed)
    np.random.seed(int(seed))
    model = {}
    functions = {'LSH': LSH, 'Fly': FlyLSH}
    sample_dim = training_data.shape[1]
    for expriment in all_expriments: #['Fly', 'FlylshDevelop', 'FlylshDevelopThreshold','FlylshDevelopThresholdRandomChoice']
        # print(expriment)
        if expriment == 'LSH':
            model[expriment] = LSH(sample_dim, hash_length) # model[expriment]变成了LSH对象，后面可以调用LSH的方法
            model[expriment].generate_weights()
            model[expriment].hashing(testing_data, center_data=if_center_data)

        else:
            #model[Fly] = FlyLSH(sample_dim, hash_length=2, sampling_ratio = 0.10, embedding_size=2*20)
            #model[FlylshDevelop] = FlyLSHDevelop(sample_dim, hash_length=2, sampling_ratio = 0.10, embedding_size=2*20)
            t0 = time.time()
            model[expriment] = functions[expriment](sample_dim, hash_length, sampling_ratio, embedding_size)
            model[expriment].generate_weights(training_data)
            model[expriment].PNtoKC(testing_data, center_data=if_center_data)
            model[expriment].KCtoMBON(testing_data, lables, center_data=if_center_data)
            t1 = time.time()
            training_time = t1 - t0
            print(training_time)

        acc = model[expriment].predict(testing_data, lables, images_nonave)
        return acc

def lifelong_test(hash_length, embedding_size, training_data, testing_data, sampling_ratio,
                all_expriments, lables, images_nonave, if_center_data=True):
    seed = hash_length * embedding_size * sampling_ratio
    random.seed(seed)
    np.random.seed(int(seed))
    model = {}
    functions = {'LSH': LSH, 'Fly': FlyLSH}
    sample_dim = training_data.shape[1]
    for expriment in all_expriments: #['Fly', 'FlylshDevelop', 'FlylshDevelopThreshold','FlylshDevelopThresholdRandomChoice']
        # print(expriment)
        if expriment == 'LSH':
            model[expriment] = LSH(sample_dim, hash_length) # model[expriment]变成了LSH对象，后面可以调用LSH的方法
            model[expriment].generate_weights()
            model[expriment].hashing(testing_data, center_data=if_center_data)

        else:
            accs1 = []
            accs2 = []
            accs3 = []
            for i in range(0, 87): #13 #87
                #acc = np.zeros((i + 1, 3))
                images = training_data[:72 * (i + 1), :]
                lables_ = lables[:72 * (i + 1)]
                model[expriment] = functions[expriment](sample_dim, hash_length, sampling_ratio, embedding_size)
                model[expriment].generate_weights(images)
                model[expriment].PNtoKC(images, center_data=if_center_data)
                model[expriment].KCtoMBON(images, lables, center_data=if_center_data)
                acc1, acc2, acc3 = model[expriment].lifelong_predict(i+1, images, lables_, images_nonave)
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
            result1 = np.zeros((87, 87))#((100, 100))#((87, 87))#((13, 13))
            for i, row in enumerate(accs1):
                result1[i][:len(row)] = row
            result2 = np.zeros((87, 87))
            for i, row in enumerate(accs2):
                result2[i][:len(row)] = row
            result3 = np.zeros((87, 87))
            for i, row in enumerate(accs3):
                result3[i][:len(row)] = row
            np.savetxt('./saved_csv/coil_del/longlife_sim_me1.csv', result1, delimiter=',')
            np.savetxt('./saved_csv/coil_del/longlife_sim_me2.csv', result2, delimiter=',')
            np.savetxt('./saved_csv/coil_del/longlife_sim_me3.csv', result3, delimiter=',')

            # with open("./saved_csv/lifelong_accs_ave.txt", "w") as file:
            #     for item in accs:
            #         file.write(str(item) + "\n")
        #np.savetxt('longlife_accs.txt', accs, delimiter=',')
        return result3

def hugeimage_test(hash_length, embedding_size, training_data, testing_data, sampling_ratio,
                all_expriments, lables, images_nonave, if_center_data=True):
    seed = hash_length * embedding_size * sampling_ratio
    random.seed(seed)
    np.random.seed(int(seed))
    model = {}
    functions = {'LSH': LSH, 'Fly': FlyLSH}
    sample_dim = training_data.shape[1]
    for expriment in all_expriments: #['Fly', 'FlylshDevelop', 'FlylshDevelopThreshold','FlylshDevelopThresholdRandomChoice']
        # print(expriment)
        if expriment == 'LSH':
            model[expriment] = LSH(sample_dim, hash_length) # model[expriment]变成了LSH对象，后面可以调用LSH的方法
            model[expriment].generate_weights()
            model[expriment].hashing(testing_data, center_data=if_center_data)

        else:
            #model[Fly] = FlyLSH(sample_dim, hash_length=2, sampling_ratio = 0.10, embedding_size=2*20)
            #model[FlylshDevelop] = FlyLSHDevelop(sample_dim, hash_length=2, sampling_ratio = 0.10, embedding_size=2*20)
            accnums = np.zeros(3)
            for i in range(0, 72001, 7200):
                images = training_data[7200*i:7200*(i+1), :]
                lables_ = lables[7200*i : 7200*(i+1)]
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
    # import tensorflow_datasets as tfds
    # dataset = tfds.load("coil100", split=tfds.Split.TRAIN, as_supervised=True)
    max_index = 10000
    sampling_ratio = 0.10
    nnn = 200  # number of nearest neighbours to compare, 2% of max_index as in paper
    hash_lengths = [ 512]#2, 32, 64, 128, 256,
    number_of_tests = 1
    ratio = 20
    result_root_path ="./Results/NoPreprocessing/" # "./Results/main/"
    lifelong = False
    hugeimge = False
    sorted_data = False
    number_of_process = 1

    threshold = 10
    if_center_data = False
    all_MAPs = {}

    # folder_path = "E:/pythonProject/data/COIL_SIM"
    # images_sim = []
    # lables = []
    # for i in range(1, 101):
    #     #if i == 2 or i == 4 or i == 25 or i == 34 or i == 35 or i == 47 or i == 53 or i == 56 or i == 73 or i == 83 or i == 86 or i == 94 or i == 95:
    #         #for j in range(0, 356, 5):
    #     # if i == 346:
    #     #     continue
    #     for j in range(0,72):
    #             I = Image.open('E:/Image_Classfiation_Coil20-master/data/coil100/coil/obj'+str(i)+'__'+str(j*5)+'.png')
    #             # #I.show()
    #             a = np.array(I)
    #             # if j==355:
    #             #     plt.imshow(a)
    #             #     plt.show()
    #             images_sim.append(a.reshape(-1))
    #             lables.append(j)
    #             print('第'+str(i)+'个图像的第'+str(j)+'个角度已加载')
    # # np.savetxt('./saved_csv/aloi_ori_96_image.csv', images_sim, delimiter=',')
    # # np.savetxt('./saved_csv/aloi_ori_96_lable.csv', lables, delimiter=',')
    # np.savetxt('./saved_csv/coil_image_ori.csv', images_sim, delimiter=',')
    # np.savetxt('./saved_csv/coil_lable_ori.csv', lables, delimiter=',')

    # images = [tf.squeeze(tf.reshape(data[0], [-1, 128*128*3])) for data in dataset] #7200*49152
    # images_num = np.array(images)
    # lables = [data[1] for data in dataset]
    # lables_num = np.array(lables)
    # np.savetxt('coil_image.csv', images, delimiter=',')
    # np.savetxt('coil_lable.csv', lables, delimiter=',')
    # images = np.loadtxt('coil_image.csv', delimiter=',')
    # images_sim = np.resize(images, (len(images), 3072))
    # np.savetxt('coil_image_sim.csv', images_sim, delimiter=',')

    # images_sim = np.loadtxt('coil_image_sim.csv', delimiter=',')
    # lables = np.loadtxt('coil_lable.csv', dtype=int, delimiter=',')
    #images_sim = images_sim_all[:72, :]
    #coil_image_del.csv

    images_sim_all = np.loadtxt('./saved_csv/coil_image_sim.csv', delimiter=',') #, max_rows=144 , max_rows=936 , max_rows=14800
    angles = np.loadtxt('./saved_csv/coil_lable_sim.csv', dtype=int, delimiter=',') #, max_rows=144
    lables_all = angles // 5 #sim

    # lables_all = angles #coil-ori aloi
    print('loaded')
    images_nonave = images_sim_all
    images_ave = (images_sim_all - np.mean(images_sim_all, axis=1)[:, None])
    #5.初始化准确率列表
    all_expriments = ['Fly']

    img_accs = [] #87
    img_nonaccs = []
    img_aveaccs = []

    if lifelong:
        for hash_length in hash_lengths:  # k
            print('life-long')
            embedding_size = int(ratio * hash_length)  # int(10*input_dim) #20k or 10d  #20*[2, 4, 8, 12, 16, 20, 24, 28, 32]
            acc = lifelong_test(hash_length, embedding_size, images_nonave, images_nonave, sampling_ratio,
                                all_expriments, lables_all, images_nonave, if_center_data)
            print(acc[:,:10])
            img_accs.append(acc)

    else:
        for hash_length in hash_lengths:
            print(hash_length)# k

            embedding_size = int(20 * hash_length)  # int(10*input_dim) #20k or 10d
            acc_nonave = single_test(hash_length, embedding_size, images_nonave, images_nonave, sampling_ratio,
                                     all_expriments, lables_all, images_nonave, if_center_data)
            acc_ave = single_test(hash_length, embedding_size, images_ave, images_ave, sampling_ratio,
                                  all_expriments, lables_all, images_nonave, if_center_data)


            img_nonaccs.append(acc_nonave)
            print(acc_ave)
            print(acc_nonave)
            img_aveaccs.append(acc_ave)
        # np.savetxt('./saved_csv/coil_sim/kcs_acc_ave.txt', img_aveaccs, delimiter=',')
        # np.savetxt('./saved_csv/coil_sim/kcs_acc_nonave.txt', img_nonaccs, delimiter=',')

