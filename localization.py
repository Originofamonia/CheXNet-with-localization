import numpy as np
from tqdm import tqdm
import skimage.transform
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import sys
import os
import pickle
import imageio
from collections import OrderedDict

import skimage
from skimage.io import *
from skimage.transform import *

import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage import binary_dilation
import matplotlib.patches as patches

from train import DenseNet121, ChestXrayDataset


# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"


# ======= Grad CAM Function =========
class PropagationBase(object):

    def __init__(self, model, cuda=False):
        self.model = model
        self.model.eval()
        if cuda:
            self.model.cuda()
        self.cuda = cuda
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()
        self._set_hook_func()
        self.image = None

    def _set_hook_func(self):
        raise NotImplementedError

    def _encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.preds.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot.cuda() if self.cuda else one_hot

    def forward(self, image):
        self.image = image
        self.preds = self.model.forward(self.image)
        #         self.probs = F.softmax(self.preds)[0]
        #         self.prob, self.idx = self.preds[0].data.sort(0, True)
        return self.preds.cpu().detach().numpy()

    def backward(self, idx):
        self.model.zero_grad()
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)


class GradCAM(PropagationBase):

    def _set_hook_func(self):

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.data.cpu()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].cpu()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.item()

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        self.map_size = grads.size()[2:]
        return nn.AvgPool2d(self.map_size)(grads)

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)
        gcam = torch.FloatTensor(self.map_size).zero_()
        for fmap, weight in zip(fmaps[0], weights[0]):
            gcam += fmap * weight.data

        gcam = F.relu(Variable(gcam))

        gcam = gcam.data.cpu().numpy()
        gcam -= gcam.min()
        gcam /= gcam.max()
        gcam = cv2.resize(gcam, (self.image.size(3), self.image.size(2)))

        return gcam

    def save(self, filename, gcam, raw_image):
        gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
        gcam = gcam.astype(np.float) + raw_image.astype(np.float)
        gcam = gcam / gcam.max() * 255.0
        cv2.imwrite(filename, np.uint8(gcam))


def main():
    cudnn.benchmark = True
    n_epochs = 10
    n_classes = 15  # has 'no finding'
    batch_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_dataset = ChestXrayDataset(train_or_test="test",
                                    transform=transforms.Compose([
                                        transforms.ToPILImage(),
                                        # transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
                                    ]))

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4)

    model = DenseNet121(n_classes).to(device)
    gcam = GradCAM(model=model, cuda=True)  # not pytorch model
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(
        "ckpt/DenseNet121_6_0.802.pkl", ))  # map_location={"cuda:0,1": 'cuda:0'}
    print("model loaded.")

    thresholds = np.load("ckpt/thresholds.npy")
    print("activate threshold: ", thresholds)

    print("generate heatmap ..........")

    # ======== Create heatmap ===========

    heatmap_output = []
    image_id = []
    output_class = []

    pbar = tqdm(test_loader)
    for i, batch in enumerate(pbar):
        batch = tuple(item.to(device) for item in batch)
        img, label, weight = batch
        probs = gcam.forward(img)  # [1, 15]
        # print(probs)

        activate_classes = np.where((probs < thresholds)[0] is True)[
            0]  # get the activated class
        print(f'activate_classes: {activate_classes}')
        for activate_class in activate_classes:
            gcam.backward(idx=activate_class)
            output = gcam.generate(
                target_layer="module.densenet121.features.denseblock4.denselayer16.conv2")
            print(f'cam output: {output}')
            # this output is heatmap
            if np.sum(np.isnan(output)) > 0:
                print("fxxx nan")
            heatmap_output.append(output)
            image_id.append(i)
            output_class.append(activate_class)
        pbar.set_description(f'Heatmap test: {i}')
    print("heatmap output done")
    print("total number of heatmap: ", len(heatmap_output))

    # ======= Plot bounding box =========
    img_width, img_height = 224, 224
    img_width_exp, img_height_exp = 1024, 1024

    crop_del = 16
    rescale_factor = 4

    # class_index = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    #                'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
    # avg_size = np.array(
    #     [[411.8, 512.5, 219.0, 139.1], [348.5, 392.3, 479.8, 381.1],
    #      [396.5, 415.8, 221.6, 318.0], [394.5, 389.1, 294.0, 297.4],
    #      [434.3, 366.7, 168.7, 189.8], [502.4, 458.7, 71.9, 70.4],
    #      [378.7, 416.7, 276.5, 304.5], [369.3, 209.4, 198.9, 246.0]])
    class_index = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                   'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
                   'Infiltration', 'Mass', 'No Finding', 'Nodule',
                   'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
    avg_size = np.array(  # xywh
        [[411.8, 512.5, 219.0, 139.1],  # Atelectasis
         [348.5, 392.3, 479.8, 381.1],  # Cardiomegaly
         [378.7, 416.7, 276.5, 304.5],  # Consolidation, no
         [369.3, 209.4, 198.9, 246.0],  # Edema, no
         [396.5, 415.8, 221.6, 318.0],  # Effusion
         [394.5, 389.1, 294.0, 297.4],  # Emphysema, no
         [434.3, 366.7, 168.7, 189.8],  # Fibrosis, no
         [502.4, 458.7, 71.9, 70.4],  # Hernia, no
         [394.5, 389.1, 294.0, 297.4],  # Infiltration
         [434.3, 366.7, 168.7, 189.8],  # Mass
         [411.8, 512.5, 219.0, 139.1],  # No Finding, no
         [502.4, 458.7, 71.9, 70.4],  # Nodule
         [396.5, 415.8, 221.6, 318.0],  # Pleural_Thickening, no
         [378.7, 416.7, 276.5, 304.5],  # Pneumonia
         [369.3, 209.4, 198.9, 246.0],  # Pneumothorax
         ])

    prediction_dict = {}
    for i in range(len(test_dataset)):
        prediction_dict[i] = []

    for img_id, k, npy in zip(image_id, output_class, heatmap_output):

        # data = npy
        # img_fname = test_dataset[img_id]

        # output average
        prediction_sent = '%s %.1f %.1f %.1f %.1f' % (
            class_index[k], avg_size[k][0], avg_size[k][1], avg_size[k][2],
            avg_size[k][3])
        prediction_dict[img_id].append(prediction_sent)

        if np.isnan(npy).any():
            continue

        w_k, h_k = (avg_size[k][2:4] * (256 / 1024)).astype(np.int)

        # Find local maxima
        neighborhood_size = 100
        threshold = 0.1

        data_max = filters.maximum_filter(npy, neighborhood_size)
        maxima = (npy == data_max)
        data_min = filters.minimum_filter(npy, neighborhood_size)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        for _ in range(5):
            maxima = binary_dilation(maxima)

        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        xy = np.array(
            ndimage.center_of_mass(npy, labeled, range(1, num_objects + 1)))

        for pt in xy:
            if npy[int(pt[0]), int(pt[1])] > np.max(npy) * .9:
                upper = int(max(pt[0] - (h_k / 2), 0.))
                left = int(max(pt[1] - (w_k / 2), 0.))

                right = int(min(left + w_k, img_width))
                lower = int(min(upper + h_k, img_height))

                prediction_sent = '%s %.1f %.1f %.1f %.1f' % (
                    class_index[k], (left + crop_del) * rescale_factor,
                    (upper + crop_del) * rescale_factor,
                    (right - left) * rescale_factor,
                    (lower - upper) * rescale_factor)

                prediction_dict[img_id].append(prediction_sent)
    img_folder_path = '/home/qiyuan/2021summer/nih/data/images'
    test_txt_path = '/home/qiyuan/2021summer/nih/data/test_list.txt'
    with open(test_txt_path, "r") as f:
        test_list = [i.strip() for i in f.readlines()]

    with open("bounding_box.txt", "w") as f:
        for i in range(len(prediction_dict)):
            fname = test_list[i]
            prediction = prediction_dict[i]

            # print(os.path.join(img_folder_path, fname), len(prediction))
            f.write('%s %d\n' % (
                os.path.join(img_folder_path, fname), len(prediction)))

            for p in prediction:
                print(p)
                f.write(p + "\n")


if __name__ == '__main__':
    main()
