import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import skimage.transform
import sys
from os import listdir
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer


def get_labels(pic_id, meta_data):
    labels = meta_data.loc[meta_data["Image Index"] == pic_id, "Finding Labels"]
    return labels.tolist()[0].split("|")


def main():
    image_folder_path = '/home/qiyuan/2021summer/nih/data/images'  # folder contain all images
    data_entry_path = '/home/qiyuan/2021summer/nih/data/Data_Entry_2017.csv'
    bbox_list_path = '/home/qiyuan/2021summer/nih/data/BBox_List_2017.csv'
    train_txt_path = '/home/qiyuan/2021summer/nih/data/train_val_list.txt'
    test_txt_path = '/home/qiyuan/2021summer/nih/data/test_list.txt'
    data_path = '/home/qiyuan/2021summer/CheXNet-with-localization/data/'  # output folder for preprocessed data

    # load data
    meta_data = pd.read_csv(data_entry_path)
    bbox_list = pd.read_csv(bbox_list_path)
    with open(train_txt_path, "r") as f:
        train_list = [i.strip() for i in f.readlines()]
    with open(test_txt_path, "r") as f:
        test_list = [i.strip() for i in f.readlines()]
    # label_eight = list(np.unique(bbox_list["Finding Label"])) + ["No Finding"]

    # process label
    print("label preprocessing")
    train_y = []
    for train_id in train_list:
        train_y.append(get_labels(train_id, meta_data))
    test_y = []
    for test_id in test_list:
        test_y.append(get_labels(test_id, meta_data))

    binarizer = MultiLabelBinarizer()
    binarizer.fit(train_y + test_y)
    train_y_onehot = binarizer.transform(train_y)
    test_y_onehot = binarizer.transform(test_y)
    train_y_onehot = np.delete(train_y_onehot, [2, 3, 5, 6, 7, 10, 12],
                               1)  # delete out 8 and "No Finding" column
    test_y_onehot = np.delete(test_y_onehot, [2, 3, 5, 6, 7, 10, 12],
                              1)  # delete out 8 and "No Finding" column

    with open(data_path + "/train_y_onehot.pkl", "wb") as f:
        pickle.dump(train_y_onehot, f)
    with open(data_path + "/test_y_onehot.pkl", "wb") as f:
        pickle.dump(test_y_onehot, f)
    with open(data_path + "/label_encoder.pkl", "wb") as f:
        pickle.dump(binarizer, f)

    # transform training images
    print(f"training example: {len(train_list)}")
    # print("take care of your RAM here !!!")
    train_x = []
    pbar = tqdm(train_list)
    for i, t in enumerate(pbar):
        image_path = os.path.join(image_folder_path, train_list[i])
        img = imageio.imread(image_path)
        if img.shape != (1024,
                         1024):  # there are some images with shape (1024,1024,4) in training set
            img = img[:, :, 0]
        img_resized = skimage.transform.resize(img, (
        256, 256))  # or use img[::4] here
        train_x.append((np.array(img_resized) / 255).reshape(256, 256, 1))
        if i % 3000 == 0:
            pbar.set_description(f'train_list: {i}/{len(train_list)}')

    train_x = np.array(train_x)
    np.save(os.path.join(data_path, "train_x_small.npy"), train_x)

    # transform validation images
    print(f"Test example: {len(test_list)}")
    test_x = []
    pbar = tqdm(test_list)
    for i, v in enumerate(pbar):
        image_path = os.path.join(image_folder_path, test_list[i])
        img = imageio.imread(image_path)
        if img.shape != (1024, 1024):
            img = img[:, :, 0]
        img_resized = skimage.transform.resize(img, (256, 256))
        #     if img.shape != (1024,1024):
        #             train_x.append(img[:,:,0])
        #     else:
        test_x.append((np.array(img_resized) / 255).reshape(256, 256, 1))
        if i % 3000 == 0:
            pbar.set_description(f'test_list: {i}/{len(test_list)}')

    test_x = np.array(test_x)
    np.save(os.path.join(data_path, "test_x_small.npy"), test_x)


if __name__ == '__main__':
    main()
