import os
import numpy as np
from tqdm import tqdm
from lshutils import Dataset, construct_true_nns, single_test, single_test_result_collector, plot_results #原始
import torch
from torch import nn
from torchvision import transforms, models, datasets
import torchvision
from PIL import Image
from squeezenet import SqueezeNet
from torchvision.models import alexnet
from torchvision.models import resnet18

if __name__ == '__main__':
    #1.一些初始化定义
    data_set_name = 'IMAGENET' #'CIFAR10'#'IMAGENET' #'SIFT' # 'CIFAR10' # 'MNIST'  #   'GLOVE'# 'CIFAR10'
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
    number_of_process = 4

    if_center_data = True

    #2.将数据集加载到inputs中
    import matplotlib.pyplot as plt
    if data_set_name =='CIFAR10':

        model = alexnet(pretrained=True)
        #model = resnet18(pretrained=True)
        #new_classifier = nn.Sequential(*list(model.children())[:5])
        new_feature = nn.Sequential(*list(model.features.children())[:11])
        new_classifier = nn.Sequential(*list(model.classifier.children())[:0])
        #print(nn.Sequential(*list(model.children())[:5]))
        print(nn.Sequential(*list(model.features.children())[:11]))

        # print(nn.Sequential(*list(model.classifier.children())))
        model.features=new_feature
        model.classifier = new_classifier
        print(nn.Sequential(*list(model.classifier.children())))

        data_transforms =transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        inputs_ori = data_set.train_batches(batch_size=max_index)
        inputs_ori = list(inputs_ori.__next__())[0]
        len_ori = len(inputs_ori)
        inputs_ = np.zeros((1000, 9216), float)
        for i in range(len_ori):
            im = inputs_ori[i, :]
            im_r = im[0:1024].reshape(32, 32)
            im_g = im[1024:2048].reshape(32, 32)
            im_b = im[2048:].reshape(32, 32)
            img = np.dstack((im_r, im_g, im_b))
            img_p = Image.fromarray(np.uint8(img))

            input_tensor = data_transforms(img_p)
            input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')
            with torch.no_grad():
                output = model(input_batch)
                #output_cpu = torch.flatten(output, 1).cpu()
                output_cpu = output.cpu()
                inputs_[i] = output_cpu.numpy()

        print()

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

        plt.imshow(inputs_[0,:].reshape(28,28))
        plt.show()

    elif data_set_name == 'IMAGENET':
        inputs_ori = np.zeros((1000, 150528), float) #原始数据长度为224*224*3
        inputs_ = np.zeros((1000, 9216), float)
        model = alexnet(pretrained=True)
        # model = resnet18(pretrained=True)
        # new_classifier = nn.Sequential(*list(model.children())[:5])
        new_feature = nn.Sequential(*list(model.features.children())[:11])
        new_classifier = nn.Sequential(*list(model.classifier.children())[:0])
        # print(nn.Sequential(*list(model.children())[:5]))
        print(nn.Sequential(*list(model.features.children())[:11]))

        # print(nn.Sequential(*list(model.classifier.children())))
        model.features = new_feature
        model.classifier = new_classifier
        print(nn.Sequential(*list(model.classifier.children())))

        data_transforms = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img_id = 0
        for img_id in range(1000):
            img = data_set.training_data[img_id][0].numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            inputs_ori[img_id] = img.reshape(-1)
            img_p = Image.fromarray(np.uint8(img))
            input_tensor = data_transforms(img_p)
            input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')
            with torch.no_grad():
                output = model(input_batch)
                # output_cpu = torch.flatten(output, 1).cpu()
                output_cpu = output.cpu()
                inputs_[img_id] = output_cpu.numpy()

       # print(inputs_)
    elif data_set_name == 'COIL':
        import cv2
        inputs_ = np.zeros((1000, 49152), float) # 原始数据长度为128*128*3
        img_id = 0
        sift_keypoints = []
        num = 0
        dirpath = 'E:/DevFly-master/datasets/COIL_SIM'
        for filename in os.listdir(dirpath):
            img = cv2.imread(dirpath + "/" + filename)
            inputs_[num] = img.reshape(-1)
            num = num + 1
        print(inputs_)
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
    all_NNs = construct_true_nns(preprocessed_data, range(inputs_ori.shape[0]), nnn, parallel=parallel,
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