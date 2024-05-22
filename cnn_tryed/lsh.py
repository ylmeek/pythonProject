import os
import numpy as np
from tqdm import tqdm
from lshutils import Dataset, construct_true_nns, single_test, single_test_result_collector, plot_results #原始
import torch
from torch import nn
from torch.nn.functional import  interpolate
from torch import optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class AlexNet(nn.Module):
    def __init__(self,num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # output[128, 6, 6]
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return x

if __name__ == '__main__':
    #1.一些初始化定义
    data_set_name = 'IMAGENET'#'FashionMNIST' #'SIFT' # 'CIFAR10' # 'MNIST'  #   'GLOVE'# 'CIFAR10'
    data_set = Dataset(data_set_name)
    input_dim = data_set.indim   # d
    max_index = 1000
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
    import matplotlib.pyplot as plt
    import cv2
    if data_set_name =='CIFAR10':
        model = AlexNet()
        preprocess = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        inputs_ori = data_set.train_batches(batch_size=max_index)
        inputs_ori = list(inputs_ori.__next__())[0]
        len_ori = len(inputs_ori)
        for i in range(len_ori):
            im = inputs_ori[0, :]
            im_r = im[0:1024].reshape(32, 32)
            im_g = im[1024:2048].reshape(32, 32)
            im_b = im[2048:].reshape(32, 32)
            img = np.dstack((im_r, im_g, im_b))

            input_tensor = preprocess(img)
            input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')
            with torch.no_grad():
                output = model(input_batch)
                inputs_ = output.tolist()

    elif data_set_name == 'IMAGENET':
        model = AlexNet()
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        inputs_ = [] #原始数据长度为224*224*3
        inputs_ori = []
        img_id = 0
        for img_id in range(1000):
            img = data_set.training_data[img_id][0].numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            img_p = Image.fromarray(np.uint8(img))
            inputs_ori.append(img.reshape(-1))
            '''
            mean=[0.485, 0.456, 0.406] #RGB
            #mean = [123.680, 116.779,103.939 ]  #RGB 图像范围0-255时
            std=[0.229, 0.224, 0.225]#来正则化
            '''
            input_tensor = preprocess(img_p)
            input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')
            with torch.no_grad():
                output = model(input_batch)
                inputs_.append(output.tolist())
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
        preprocessed_data = (inputs_ori - np.mean(inputs_ori, axis=1)[:, None])
    else:
        preprocessed_data = inputs_ori

    #4.记录图像之间真实的距离，便于后面计算准确率等
    all_NNs = construct_true_nns(preprocessed_data, range(len(inputs_ori)), nnn, parallel=parallel,
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
                    [hash_length, i, embedding_size, inputs_, inputs_, sampling_ratio, nnn, all_NNs, all_expriments,
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
                single_test_result_collector(hash_length, test_index, embedding_size, inputs_, inputs_, sampling_ratio, nnn,
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