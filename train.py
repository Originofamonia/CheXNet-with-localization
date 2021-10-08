import os
import pickle
import sys

import numpy as np
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"


def compute_AUCs(gt, pred, N_CLASSES):
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


# ====== prepare dataset ======
class ChestXrayDataSet(Dataset):
    def __init__(self, train_or_test="train", transform=None):

        data_path = '/home/qiyuan/2021summer/CheXNet-with-localization/data/'
        self.train_or_valid = train_or_test
        if train_or_test == "train":
            self.X = np.uint8(
                np.load(data_path + "train_x_small.npy") * 255 * 255)
            with open(data_path + "train_y_onehot.pkl", "rb") as f:
                self.y = pickle.load(f)
            sub_bool = (self.y.sum(axis=1) != 0)
            self.y = self.y[sub_bool, :]
            self.X = self.X[sub_bool, :]
        else:
            self.X = np.uint8(
                np.load(data_path + "test_x_small.npy") * 255 * 255)
            with open(data_path + "test_y_onehot.pkl", "rb") as f:
                self.y = pickle.load(f)

        self.label_weight_pos = (len(self.y) - self.y.sum(axis=0)) / len(self.y)
        self.label_weight_neg = (self.y.sum(axis=0)) / len(self.y)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item 
        Returns:
            img and its labels
        """
        img = np.tile(self.X[index], 3)
        label = self.y[index]
        label_inverse = 1 - label
        weight = np.add((label_inverse * self.label_weight_neg),
                        (label * self.label_weight_pos))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.from_numpy(label).type(
            torch.FloatTensor), torch.from_numpy(weight).type(torch.FloatTensor)

    def __len__(self):
        return len(self.y)


# construct model
class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


def main():
    cudnn.benchmark = True
    n_epochs = 10
    n_classes = 15  # has 'no finding'
    BATCH_SIZE = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # prepare training set
    train_dataset = ChestXrayDataSet(train_or_test="train",
                                     transform=transforms.Compose([
                                         transforms.ToPILImage(),
                                         transforms.CenterCrop(224),
                                         # was RandomCrop
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             [0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225]),
                                     ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4)

    # prepare validation set
    test_dataset = ChestXrayDataSet(train_or_test="test",
                                    transform=transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
                                    ]))

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4)
    # ====== start training =======
    # initialize and load the model
    model = DenseNet121(n_classes).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        print("Epoch:", epoch)
        train_loss = 0.0
        model.train()
        pbar = tqdm(train_loader)
        for i, batch in enumerate(pbar):
            batch = tuple(item.to(device) for item in batch)
            img, label, weight = batch
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_description(f'Epoch: {epoch}/{n_epochs}; loss: '
                                 f'{loss.item():.3f}')
        # ======== validation ========
        # switch to evaluate mode
        model.eval()

        # initialize the ground truth and output tensor
        gt = torch.FloatTensor().to(device)
        pred = torch.FloatTensor().to(device)
        pbar = tqdm(test_loader)
        for i, batch in enumerate(pbar):
            batch = tuple(item.to(device) for item in batch)
            img, target, weight = batch
            gt = torch.cat((gt, target), 0)
            #     bs, n_crops, c, h, w = img.size()
            # input_var = Variable(img.view(-1, 3, 224, 224).cuda(), volatile=True)
            output = model(img)
            #     output_mean = output.view(bs, n_crops, -1).mean(1)
            pred = torch.cat((pred, output.data), 0)
            pbar.set_description(f'Test: {i}')

        # names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        #                'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']
        names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
                 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening',
                 'Hernia', 'No finding']  # needs changing

        AUROCs = compute_AUCs(gt, pred, n_classes)
        AUROC_avg = np.array(AUROCs).mean()
        print(f'The average AUROC is {AUROC_avg:.3f}')
        for i in range(n_classes):
            print(f'The AUROC of {names[i]} is {AUROCs[i]}')

        # print statistics
        print(f'[{epoch}] loss: {train_loss / 715:.3f}%')

    torch.save(model.state_dict(),
               f'ckpt/DenseNet121_{n_epochs}_{AUROC_avg:.3f}.pkl')

    print('Finished Training')


if __name__ == '__main__':
    main()
