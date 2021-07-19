import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import skimage.transform
import sys
from os import listdir
from sklearn.preprocessing import MultiLabelBinarizer


def get_labels(pic_id, meta_data):
    labels = meta_data.loc[meta_data["Image Index"] == pic_id, "Finding Labels"]
    return labels.tolist()[0].split("|")


def main():
    image_folder_path = '/home/qiyuan/2021summer/nih/data/images'  # folder contain all images
    data_entry_path = '/home/qiyuan/2021summer/nih/data/Data_Entry_2017.csv'
    bbox_list_path = '/home/qiyuan/2021summer/nih/data/BBox_List_2017.csv'
    train_txt_path = '/home/qiyuan/2021summer/nih/data/train_val_list.txt'
    valid_txt_path = '/home/qiyuan/2021summer/nih/data/train_val_list.txt'
    data_path = '/home/qiyuan/2021summer/CheXNet-with-localization/data'  # output folder for preprocessed data

    # load data
    meta_data = pd.read_csv(data_entry_path)
    bbox_list = pd.read_csv(bbox_list_path)
    with open(train_txt_path, "r") as f:
        train_list = [i.strip() for i in f.readlines()]
    with open(valid_txt_path, "r") as f:
        valid_list = [i.strip() for i in f.readlines()]
    # label_eight = list(np.unique(bbox_list["Finding Label"])) + ["No Finding"]

    # transform training images
    # print("training example:", len(train_list))
    # print("take care of your RAM here !!!")
    # train_X = []
    # for i in range(len(train_list)):
    #     image_path = os.path.join(image_folder_path, train_list[i])
    #     img = imageio.imread(image_path)
    #     if img.shape != (1024, 1024):  # there some image with shape (1024,1024,4) in training set
    #         img = img[:, :, 0]
    #     img_resized = skimage.transform.resize(img, (256, 256))  # or use img[::4] here
    #     train_X.append((np.array(img_resized) / 255).reshape(256, 256, 1))
    #     if i % 3000 == 0:
    #         print(i)
    #
    # train_X = np.array(train_X)
    # np.save(os.path.join(data_path, "train_X_small.npy"), train_X)

    # transform validation images
    # print("validation example:", len(valid_list))
    # valid_X = []
    # for i in range(len(valid_list)):
    #     image_path = os.path.join(image_folder_path, valid_list[i])
    #     img = imageio.imread(image_path)
    #     if img.shape != (1024, 1024):
    #         img = img[:, :, 0]
    #     img_resized = skimage.transform.resize(img, (256, 256))
    #     #     if img.shape != (1024,1024):
    #     #             train_X.append(img[:,:,0])
    #     #     else:
    #     valid_X.append((np.array(img_resized) / 255).reshape(256, 256, 1))
    #     if i % 3000 == 0:
    #         print(i)
    #
    # valid_X = np.array(valid_X)
    # np.save(os.path.join(data_path, "valid_X_small.npy"), train_X)

    # process label
    print("label preprocessing")

    train_y = []
    for train_id in train_list:
        train_y.append(get_labels(train_id, meta_data))
    valid_y = []
    for valid_id in valid_list:
        valid_y.append(get_labels(valid_id, meta_data))

    encoder = MultiLabelBinarizer()
    encoder.fit(train_y + valid_y)
    train_y_onehot = encoder.transform(train_y)
    valid_y_onehot = encoder.transform(valid_y)
    train_y_onehot = np.delete(train_y_onehot, [2, 3, 5, 6, 7, 10, 12], 1)  # delete out 8 and "No Finding" column
    valid_y_onehot = np.delete(valid_y_onehot, [2, 3, 5, 6, 7, 10, 12], 1)  # delete out 8 and "No Finding" column

    with open(data_path + "/train_y_onehot.pkl", "wb") as f:
        pickle.dump(train_y_onehot, f)
    with open(data_path + "/valid_y_onehot.pkl", "wb") as f:
        pickle.dump(valid_y_onehot, f)
    with open(data_path + "/label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)


if __name__ == '__main__':
    main()
