import os
import shutil
import torch
from torch.utils.data import random_split
import glob

def dataset_split(dataset_root, lengths, output='./', random_seed=0):
    json_path_list = glob.glob(os.path.join(dataset_root, '*.json'))
    img_path_list = glob.glob(os.path.join(dataset_root, '*.jpg'))
    # images_root = './img'
    # gt_root = './gt'
    # # 所有图片的文件名
    # img_list = os.listdir(images_root)
    #
    # # 去掉后缀的文件名list
    # filename_list = []
    # for item in img_list:
    #     filename_list.append(item.split('.')[0])
    #
    # # 划分数据集
    # torch.manual_seed(0)
    # train_dataset, test_dataset = random_split(
    #     dataset = filename_list,
    #     lengths = [500, 200],
    # )
    #
    # if not os.path.exists('./train_data'):
    #     os.mkdir('./train_data')
    #     os.mkdir('./train_data/images')
    #     os.mkdir('./train_data/ground_truth')
    # if not os.path.exists('./test_data'):
    #     os.mkdir('./test_data')
    #     os.mkdir('./test_data/images')
    #     os.mkdir('./test_data/ground_truth')
    #
    # # 复制对应文件到train文件夹
    # for item in train_dataset:
    #     # 复制图片
    #     shutil.copy(os.path.join('./img', item + '.jpg'), os.path.join('./train_data/images', item + '.jpg'))
    #     # 复制json
    #     shutil.copy(os.path.join('./gt', item + '.json'), os.path.join('./train_data/ground_truth', item + '.json'))
    #     #  复制txt
    #     shutil.copy(os.path.join('./gt', item + '.txt'), os.path.join('./train_data/ground_truth', item + '.txt'))
    #
    # # 复制对应文件到test文件夹
    # for item in test_dataset:
    #     # 复制图片
    #     shutil.copy(os.path.join('./img', item + '.jpg'), os.path.join('./test_data/images', item + '.jpg'))
    #     # 复制json
    #     shutil.copy(os.path.join('./gt', item + '.json'), os.path.join('./test_data/ground_truth', item + '.json'))
    #     #  复制txt
    #     shutil.copy(os.path.join('./gt', item + '.txt'), os.path.join('./test_data/ground_truth', item + '.txt'))
    # print(list(train_dataset))
    # print(list(test_dataset))

if __name__ == '__main':
    dataset_split(dataset_root='pig_200')
