import cv2
import torch
import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json


def xywh2xyxy(x):
    """
    Convert nx4 boxes from [x, y, w, h], xy are top-left
    https://mipt-oulu.github.io/solt/Medical_Data_Augmentation_CXR14.html
    to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0]  # top left x
    y[:, 1] = x[:, 1]  # top left y
    y[:, 2] = x[:, 0] + x[:, 2]  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3]  # bottom right y
    return y


def color_list():
    # Return first 10 plt colors as (r,g,b)
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in
            plt.rcParams['axes.prop_cycle'].by_key()['color']]


colors = color_list()  # list of colors


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255],
                    thickness=tf, lineType=cv2.LINE_AA)


def plot_image(image, boxes, findings, paths=None, fname='images.jpg',
               names=None, max_size=1024, max_subplots=16):
    # Plot image grid with labels

    # if isinstance(image, torch.Tensor):
    #     image = image.cpu().float().numpy()
    # for k, v in targets.items():
    #     if isinstance(v, torch.Tensor):
    #         targets[k] = v.cpu().numpy()

    # un-normalise
    if np.max(image) <= 1:
        image *= 255

    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    h, w, ch = image.shape  # height, width, ch
    bs = 1
    # bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    # for i, img in enumerate(image):
    #     if i == max_subplots:  # if last batch has fewer images than we expect
    #         break

    # block_x = int(w * (i // ns))
    # block_y = int(h * (i % ns))

    if scale_factor < 1:
        image = cv2.resize(image, (w, h))

    mosaic = image
    # if len(targets) > 0:
    # image_targets = targets[targets[:, 0] == i]
    # boxes = targets['boxes']
    # classes = targets['labels']  # - 1
    # labels = [names[class_id] for class_id in
    #           classes]  # labels if no conf column
    conf = None  # check for confidence presence (label vs pred)

    if boxes.shape[0]:
        # absolute coords need scale if image scales
        boxes *= scale_factor
    # boxes[[0, 2]] += block_x
    # boxes[[1, 3]] += block_y
    for j, box in enumerate(boxes):
        cls = names.index(findings[j])
        color = colors[cls % len(colors)]
        cls = names[cls] if names else cls
        if len(findings) > 0:
            label = '%s' % cls
            plot_one_box(box, mosaic, label=label, color=color,
                         line_thickness=tl)

    # Draw image filename labels
    # if paths:
    #     label = Path(paths[i]).name[:40]  # trim to 40 char
    #     t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    #     cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
    #                 lineType=cv2.LINE_AA)

    # Image border
    # cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname:
        r = min(1280. / max(h, w) / ns, 1.0)  # ratio to limit image size
        mosaic = cv2.resize(mosaic, (int(ns * w * r), int(ns * h * r)),
                            interpolation=cv2.INTER_AREA).astype(np.uint8)
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))  # cv2 save
        # Image.fromarray(mosaic).save(fname)  # PIL save
    return mosaic


def plot_bbox(img_folder_path, bbox_filename):
    """
    https://github.com/thtang/CheXNet-with-localization/issues/9
    """
    actual_bbox = open(bbox_filename)
    img_folder_path = os.path.split(img_folder_path)[-1]
    # print(img_folder_path)
    count = 0
    temp_count = 0
    final_bbox_list = []
    for img in actual_bbox:
        if img.find(img_folder_path) != -1:
            print('file exist:', count)
            print('given image', img)
            temp_count = count
            print("this is temp count", temp_count)
        if count > temp_count:

            if img.find('/') == -1:
                final_bbox_list.append(img)
            else:
                break
        count += 1

    i = final_bbox_list[1]
    temp_i = list(i.split(" "))
    temp_i.pop(0)

    p = np.array(temp_i)
    k = p.astype(float)

    x1 = int(k[0])
    y1 = int(k[1])
    x2 = int(k[2])
    y2 = int(k[3])
    return x1, y1, x2, y2


def main():
    """
    https://github.com/thtang/CheXNet-with-localization/issues/9
    """
    img_folder_path = '/media/sadam/44C611803928605D/Downloads/ChestXray-NIHCC/a/00000003_000.png'
    frame = cv2.imread(img_folder_path)

    ac_bbox = '/home/sadam/CheXNet-with-localization/output/bounding_box.txt'
    pd_bbox = '/home/sadam/CheXNet-with-localization/bounding_box.txt'

    x1, y1, x2, y2 = plot_bbox(img_folder_path, ac_bbox)
    x_1, y_1, x_2, y_2 = plot_bbox(img_folder_path, pd_bbox)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # rgb 220,20,60
    cv2.rectangle(frame, (x_1, y_1), (x_2, y_2), (60, 20, 220), 3)
    print(frame)
    cv2.imshow('image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_inferred_boxes():
    """
    read in box from txt, read image from nih path
    only plot inferred boxes in this function
    """
    names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
             'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
             'Infiltration', 'Mass', 'No Finding', 'Nodule',
             'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
    test_txt_path = '/home/qiyuan/2021summer/nih/data/test_list.txt'
    with open(test_txt_path, "r") as f:
        test_list = [i.strip() for i in f.readlines()]
    nih_folder_path = '/home/qiyuan/2021summer/nih/data'
    save_dir = 'output'
    # was xywh, xy are center; new are xyxy
    bbox_file = '/home/qiyuan/2021summer/CheXNet-with-localization/pred_boxes.json'
    df = pd.read_csv(os.path.join(nih_folder_path, 'BBox_List_2017.csv'))
    img_indices = df['Image Index'].unique()
    with open(bbox_file, 'r') as f:
        data = json.load(f)  # json box is xyxy
        for i, (k, v) in enumerate(data.items()):
            img_id = test_list[int(k)]
            findings = [item.split(' ')[0] for item in v]
            boxes = [item.split(' ')[1:] for item in v]
            boxes = np.array(boxes)
            if img_id in img_indices:
                img = cv2.imread(img_path)  # [1024, 1024, 3]
                box_lines = lines[i + 1: i + 1 + int(n_box)]
                boxes = []
                findings = []
                for j, row in enumerate(box_lines):
                    boxes.append(
                        [float(item) for item in row.split(' ')[1:]])
                    boxes = np.array(boxes)
                    findings.append(row.split(' ')[0])
                # print(findings)

                f = f'{save_dir}/{img_id.strip(".png")}_pred.jpg'  # labels
                plot_image(img, boxes, findings, None, f, names)


def plot_labels():
    """
    plot boxes from BBox_List_2017.csv
    """
    names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
             'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
             'Infiltrate', 'Mass', 'No Finding', 'Nodule',
             'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
    save_dir = 'output'
    nih_folder_path = '/home/qiyuan/2021summer/nih/data'
    df = pd.read_csv(os.path.join(nih_folder_path, 'BBox_List_2017.csv'))
    img_indices = df['Image Index'].unique()

    for img_index in img_indices:
        boxes = df.loc[df['Image Index'] == img_index, ['Bbox [x', 'y', 'w', 'h]']].values
        boxes = xywh2xyxy(boxes)
        findings = df.loc[df['Image Index'] == img_index, 'Finding Label'].values
        img = cv2.imread(os.path.join(nih_folder_path, 'images', img_index))
        f = f'{save_dir}/{img_index.strip(".png")}_label.jpg'  # labels
        plot_image(img, boxes, findings, None, f, names)


if __name__ == '__main__':
    # main()
    plot_inferred_boxes()
    # plot_labels()
