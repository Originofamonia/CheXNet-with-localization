import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
from collections import OrderedDict

import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.ndimage import binary_dilation
import json

from train import DenseNet121, ChestXrayDataset
from localization import GradCAM


def main():
    cudnn.benchmark = True
    n_classes = 15  # has 'no_finding'
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

    model.load_state_dict(torch.load(
        "ckpt/DenseNet121_6_0.802.pkl", ))  # map_location={"cuda:0,1": 'cuda:0'}
    print("Model loaded.")
    gcam = GradCAM(model=model, cuda=True)  # not pytorch model

    thresholds = np.load("ckpt/thresholds.npy")
    print("Activate threshold: ", thresholds)

    print("Generate heatmap ..........")
    heatmap_output = OrderedDict()
    output_class = OrderedDict()

    pbar = tqdm(test_loader)
    for i, batch in enumerate(pbar):
        batch, img_id = batch[:3], batch[-1]
        batch = tuple(item.to(device) for item in batch)
        img, label, weight = batch
        probs = gcam.forward(img)  # [1, 15]
        # print(probs)

        activate_classes = np.where((probs > thresholds)[0] == True)[
            0]  # get the activated class, don't change ==
        # print(f'activate_classes: {activate_classes}')
        for acti_class in activate_classes:
            gcam.backward(idx=acti_class)
            fmap = gcam._find(gcam.all_fmaps, target_layer="densenet121.features.denseblock4.denselayer16.conv2")
            output = gcam.generate(  # add module if multi-gpu
                target_layer="densenet121.features.denseblock4.denselayer16.conv2")
            # print(f'cam output: {output}')
            # this output is heatmap
            if np.sum(np.isnan(output)) > 0:
                print(f"heatmap {img_id} has nan")
            heatmap_output[img_id] = output
            output_class[img_id] = acti_class
        pbar.set_description(f'Heatmap test: {img_id}')
    print("Heatmap output done")
    print("Total number of heatmap: ", len(heatmap_output))
    # with open('heatmaps.json', 'w') as f:
    #     json.dump(heatmap_output, f)


if __name__ == '__main__':
    main()
