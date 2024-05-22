import os
import numpy as np
from tqdm import tqdm
from lshutils import Dataset, construct_true_nns, single_test, single_test_result_collector, plot_results #原始
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import torch
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image

if __name__ == '__main__':
    #1.一些初始化定义
    data_set_name = 'IMAGENET' #'IMAGENET'#'COIL'#'FashionMNIST' #'SIFT' # 'CIFAR10' # 'MNIST'  #   'GLOVE'# 'CIFAR10'
    data_set = Dataset(data_set_name)
    input_dim = data_set.indim   # d
    max_index = 10000
    sampling_ratio = 0.10
    nnn = 200  # number of nearest neighbours to compare, 2% of max_index as in paper
    hash_lengths = [2, 4, 8, 12, 16, 20, 24, 28, 32]
    number_of_tests = 10
    ratio = 20
    sorted_data = False
    result_root_path ="./Results/NoPreprocessing/" # "./Results/main/"
    parallel = True
    number_of_process = 5

    if_center_data = True

    #2.将数据集加载到inputs中


    if data_set_name =='CIFAR10':
        class Inception(nn.Module):
            # c1 - c4为每条线路里的层的输出通道数
            def __init__(self, in_c, c1, c2, c3, c4):
                super(Inception, self).__init__()
                # 线路1，单1 x 1卷积层
                self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
                # 线路2，1 x 1卷积层后接3 x 3卷积层
                self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
                self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
                # 线路3，1 x 1卷积层后接5 x 5卷积层
                self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
                self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
                # 线路4，3 x 3最大池化层后接1 x 1卷积层
                self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
                self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

            def forward(self, x):
                p1 = F.relu(self.p1_1(x))
                p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
                p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
                p4 = F.relu(self.p4_2(self.p4_1(x)))
                return torch.cat((p1, p2, p3, p4), dim=1)  # 在通道维上连结输出
        class GlobalAvgPool2d(nn.Module):
            # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
            def __init__(self):
                super(GlobalAvgPool2d, self).__init__()

            def forward(self, x):
                return F.avg_pool2d(x, kernel_size=x.size()[2:])
        class FlattenLayer(torch.nn.Module):
            def __init__(self):
                super(FlattenLayer, self).__init__()

            def forward(self, x):  # x shape: (batch, *, *, ...)
                return x.view(x.shape[0], -1)
        class GoogLeNet(nn.Module):
            def __init__(self, num_classes=1000):
                super(GoogLeNet, self).__init__()

                self.b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

                self.b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                                        nn.Conv2d(64, 192, kernel_size=3, padding=1),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

                self.b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                                        Inception(256, 128, (128, 192), (32, 96), 64),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

                self.b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                                        Inception(512, 160, (112, 224), (24, 64), 64),
                                        Inception(512, 128, (128, 256), (24, 64), 64),
                                        Inception(512, 112, (144, 288), (32, 64), 64),
                                        Inception(528, 256, (160, 320), (32, 128), 128),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

                self.b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                                        Inception(832, 384, (192, 384), (48, 128), 128),
                                        GlobalAvgPool2d())
                self.output = nn.Sequential(FlattenLayer(),
                                            nn.Dropout(p=0.4),
                                            nn.Linear(1024, 1000))

            def forward(self, x):
                x = self.b1(x)
                x = self.b2(x)
                x = self.b3(x)
                x = self.b4(x)
                x = self.b5(x)
                x = self.output(x)
                return x

        model = GoogLeNet()
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        inputs_ori = data_set.train_batches(batch_size=max_index)
        inputs_ori = list(inputs_ori.__next__())[0]
        inputs_ = []
        inputs_net = []
        len_ori = len(inputs_ori)

        for i in range(1000):
            im = inputs_ori[i, :]
            im_r = im[0:1024].reshape(32, 32)
            im_g = im[1024:2048].reshape(32, 32)
            im_b = im[2048:].reshape(32, 32)
            img = np.dstack((im_r, im_g, im_b))
            img_p = Image.fromarray(np.uint8(img))
            inputs_.append(im)

            input_tensor = preprocess(img_p)
            input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')
            with torch.no_grad():
                output = model(input_batch)
                inputs_net.append(output.tolist())


    elif data_set_name == 'IMAGENET':
        class Inception(nn.Module):
            # c1 - c4为每条线路里的层的输出通道数
            def __init__(self, in_c, c1, c2, c3, c4):
                super(Inception, self).__init__()
                # 线路1，单1 x 1卷积层
                self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
                # 线路2，1 x 1卷积层后接3 x 3卷积层
                self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
                self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
                # 线路3，1 x 1卷积层后接5 x 5卷积层
                self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
                self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
                # 线路4，3 x 3最大池化层后接1 x 1卷积层
                self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
                self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

            def forward(self, x):
                p1 = F.relu(self.p1_1(x))
                p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
                p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
                p4 = F.relu(self.p4_2(self.p4_1(x)))
                return torch.cat((p1, p2, p3, p4), dim=1)  # 在通道维上连结输出
        class GlobalAvgPool2d(nn.Module):
            # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
            def __init__(self):
                super(GlobalAvgPool2d, self).__init__()

            def forward(self, x):
                return F.avg_pool2d(x, kernel_size=x.size()[2:])
        class FlattenLayer(torch.nn.Module):
            def __init__(self):
                super(FlattenLayer, self).__init__()

            def forward(self, x):  # x shape: (batch, *, *, ...)
                return x.view(x.shape[0], -1)
        class GoogLeNet(nn.Module):
            def __init__(self, num_classes=1000):
                super(GoogLeNet, self).__init__()

                self.b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

                self.b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                                        nn.Conv2d(64, 192, kernel_size=3, padding=1),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

                self.b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                                        Inception(256, 128, (128, 192), (32, 96), 64),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

                self.b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                                        Inception(512, 160, (112, 224), (24, 64), 64),
                                        Inception(512, 128, (128, 256), (24, 64), 64),
                                        Inception(512, 112, (144, 288), (32, 64), 64),
                                        Inception(528, 256, (160, 320), (32, 128), 128),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

                self.b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                                        Inception(832, 384, (192, 384), (48, 128), 128),
                                        GlobalAvgPool2d())
                self.output = nn.Sequential(FlattenLayer(),
                                            nn.Dropout(p=0.4),
                                            nn.Linear(1024, 1000))

            def forward(self, x):
                x = self.b1(x)
                x = self.b2(x)
                x = self.b3(x)
                x = self.b4(x)
                x = self.b5(x)
                x = self.output(x)
                return x

        model = GoogLeNet()
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        inputs_ = [] #原始数据长度为224*224*3
        inputs_net = []
        img_id = 0
        for img_id in range(1000):
            img = data_set.training_data[img_id][0].numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            img_p = Image.fromarray(np.uint8(img))
            inputs_.append(img.reshape(-1))

            input_tensor = preprocess(img_p)
            input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')
            with torch.no_grad():
                output = model(input_batch)
                inputs_net.append(output.tolist())

    elif data_set_name == 'COIL':
        inputs_ = np.zeros((7200, 49152), float)  # 原始数据长度为128*128*3
        img_id = 0
        sift_keypoints = []
        num = 0
        dirpath = 'E:/DevFly-master/datasets/COIL'
        for filename in os.listdir(dirpath):
            img = cv2.imread(dirpath + "/" + filename)

            sift = cv2.SIFT_create()
            keypoint, descriptor = sift.detectAndCompute(img, None)
            if len(keypoint) >= 1:
                kp = np.array([p.pt for p in keypoint])
                sift_keypoints.append(kp)
                inputs_[num] = img.reshape(-1)
                num = num + 1

    elif data_set_name == 'FMNIST' or data_set_name == 'MNIST':
        inputs_ = np.array(data_set.training_data['images'][:max_index])
        labels_ = data_set.training_data['labels'][:max_index]
        if sorted_data:
            sorted_idx = list(np.argsort(labels_))
            inputs_ = inputs_[sorted_idx]
            labels_ = np.squeeze(labels_)[sorted_idx]

    elif data_set_name == 'FashionMNIST':
        inputs_ = np.array(data_set.training_data['images'][:max_index])
        labels_ = data_set.training_data['labels'][:max_index]
        if sorted_data:
            sorted_idx = list(np.argsort(labels_))
            inputs_ = inputs_[sorted_idx]
            labels_ = np.squeeze(labels_)[sorted_idx]

    elif data_set_name == 'SIFT':
        inputs_ = data_set.data[:max_index]
    elif data_set_name == 'GLOVE':
        inputs_ = np.array(list(data_set.data.values())[:max_index])

    plt.pause(.001)
    if data_set_name == 'GLOVE':
        threshold = 1
    else:
        threshold = 10

    all_MAPs = {}
    #3.一些归一化操作
    if if_center_data:
        preprocessed_data = (inputs_ - np.mean(inputs_, axis=1)[:, None])
    else:
        preprocessed_data = inputs_

    #4.记录图像之间真实的距离，便于后面计算准确率等
    all_NNs = construct_true_nns(preprocessed_data, range(len(inputs_)), nnn, parallel=parallel,
                                     number_of_process=number_of_process)
    print('All nearest neighbours are found')


    #5.初始化准确率列表
    all_expriments = ['Fly', 'FlylshDevelop', 'FlylshDevelopThreshold','FlylshDevelopThresholdRandomChoice'] # 'LSH',
    for hash_length in hash_lengths:  # k=[2, 4, 8, 12, 16, 20, 24, 28, 32]
        all_MAPs[hash_length] = {}
        for a_key in all_expriments: #Fly等4个方法
            all_MAPs[hash_length][a_key] = [0 for _ in range(number_of_tests)]

    #6.训练数据及计算准确率
    if parallel:
        import multiprocessing
        cup_number = multiprocessing.cpu_count()
        for hash_length in hash_lengths:  # k
            embedding_size = int(ratio * hash_length)  # int(10*input_dim) #20k or 10d  #20*[2, 4, 8, 12, 16, 20, 24, 28, 32]
            with multiprocessing.Pool(processes=number_of_process) as pool:
                results = pool.starmap(single_test, (
                    [hash_length, i, embedding_size, inputs_net, inputs_net, sampling_ratio, nnn, all_NNs, all_expriments,
                     threshold, if_center_data]
                    for i in range(number_of_tests)))
            for i in range(number_of_tests):
                for a_key in all_expriments:
                    # if results[i].has_key[a_key]:
                    all_MAPs[hash_length][a_key][i] = results[i][a_key]
    else:
        for hash_length in hash_lengths:  # k
            embedding_size = int(20 * hash_length)  # int(10*input_dim) #20k or 10d
            for test_index in tqdm(range(number_of_tests)): #每种情况下测试10次，针对每种情况的每个测试，计算准确率
                single_test_result_collector(hash_length, test_index, embedding_size, inputs_net, inputs_net, sampling_ratio, nnn,
                                             all_NNs, all_expriments, all_MAPs, if_center_data)

    print(all_MAPs)

    #画图及文件的存储
    import pickle

    result_path = os.path.join(result_root_path, data_set_name, str(sampling_ratio), str(ratio), str(sorted_data))
    print(result_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(os.path.join(result_path, 'all_MAPs'+str(max_index)+'.pkl'), 'wb') as record_file:
        pickle.dump(all_MAPs, record_file)
    pass

    with open(os.path.join(result_path, 'all_MAPs'+str(max_index)+'.pkl'), 'rb') as record_file:
        all_MAPs = pickle.load(record_file)
    figure_bokeh = plot_results(all_MAPs, plot_width=1600, plot_height=800)
    pass