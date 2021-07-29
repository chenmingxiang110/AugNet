import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-i",
                           "https://pypi.tuna.tsinghua.edu.cn/simple", "-U", package])

install("imgaug")

import os
import cv2
import time
import random
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange
from scipy.spatial import distance
from collections import OrderedDict
from imgaug import augmenters as iaa
from IPython.display import clear_output

from lib.utils_torch import Identity
from lib.utils import normalize, read_all_imgs, extractor

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

_root = "./imagenet/ILSVRC/Data/CLS-LOC/train"
model_path = "./models/final_res50_model_training_with_ImageNet.pth"

class Net(nn.Module):

    def __init__(self, path_pre=None, gpus=[]):
        super(Net, self).__init__()
        self.pre = models.resnet50(pretrained=False)

        if path_pre is not None:
            self.pre.load_state_dict(torch.load(path_pre))

        self.pre.fc = nn.Linear(2048, 768)
        self.act = nn.Tanh()

        if len(gpus) > 1:
            self.pre = nn.DataParallel(self.pre, gpus)

    def forward(self, x):
        h = self.pre(x)
        h = self.act(h)
        return h

class Loss(nn.Module):

    def __init__(self, device=torch.device("cpu")):
        super(Loss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

        self.device = device
        self.to(device)

    def forward(self, x, mode, optimizer=None, scheduler=None):
        """
        x: (k/j (classes), m/i (samples), c (n_features))
        """
        centers = torch.mean(x, 1, keepdim=False)
        sims = -torch.cdist(x.reshape([-1, x.size(-1)]), centers, p=2)
#         sims = torch.max(sims, torch.tensor(-5.0).to(self.device))

        if mode=="softmax":
            labels = torch.tensor(list(np.arange(x.size(0)).repeat(x.size(1)))).to(self.device)
            loss = self.ce(sims, labels)

        elif mode=="contrast":
            indices = list(np.arange(sims.size(0)))
            labels = list(np.arange(x.size(0)).repeat(x.size(1)))
            sims_clone = torch.clone(sims)
            sims_clone[indices, labels] = -1e32
            loss_self = sims[indices, labels]
            loss_others = torch.max(sims_clone, dim=1).values
            # loss = 1-torch.sigmoid(loss_self)+torch.sigmoid(loss_others)
            loss = loss_others-loss_self
            loss = torch.mean(loss)

        else:
            raise ValueError("Invalid mode.")

        if optimizer is not None:
            # back propagation and update centers
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        return loss

roots = sorted([x for x in os.listdir(_root) if x[0]!='.'])

paths = []
path_labels = []
for i in trange(len(roots)):
    root = os.path.join(_root, roots[i])
    paths_ = read_all_imgs(root, _iter=False)
    paths = paths+paths_
    path_labels = path_labels+[i,]*len(paths_)

paths = [x for x in zip(paths, path_labels)]
print(len(paths))

lr = 2e-4
gpus = [0,1,2,3]
n_pics = 16*len(gpus)
n_samples = 16 # min=4

if len(gpus)>0:
    device = torch.device("cuda:"+str(gpus[0]))
else:
    device = torch.device("cpu")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

etr = extractor()

my_model = Net().to(device)
my_model = nn.DataParallel(my_model, gpus)

# my_model.load_state_dict(torch.load(model_path))

optimizer = optim.Adam(my_model.parameters(), lr)
criterion = Loss(device)

best_loss = 1e8
i_epoch = -1
loss_log = []

aug_seq_all = iaa.Sequential([
    iaa.Affine(rotate=(-35, 35)),
    iaa.Crop(percent=(0, 0.1)),
    iaa.GaussianBlur(sigma=(0.0, 3.0)),
    iaa.AddToHue((-20, 20)),
    iaa.AddToSaturation((-150, 50)),
    iaa.MultiplyAndAddToBrightness(mul=(0.75, 1.3), add=(-40, 40)),
    iaa.GammaContrast((0.75, 1.3)),
    iaa.GammaContrast((0.75, 1.3), per_channel=True),
#     iaa.ChangeColorTemperature((3000, 9000)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)),
    iaa.JpegCompression(compression=(0, 99)),
    iaa.MotionBlur(k=(3,15)),
    iaa.CoarseDropout(0.02, size_percent=0.1, per_channel=0.5),
])

start_time = time.time()

while True:
    i_epoch+=1
    _indices = np.random.choice(len(paths), len(paths), replace=False)
    n_round = int(len(paths)/n_pics)
    for i_round in range(n_round):
        _augs = []
        indices = _indices[i_round * n_pics : (i_round+1) * n_pics]
        for index in indices:
            img = cv2.imread(paths[index][0])[...,::-1]
            augs = etr.extract(img, n_augs=n_samples, target_size=224, aug_seq=aug_seq_all, resolution_aug="False")
            augs = normalize(augs.transpose([0,3,1,2]), mean=mean, std=std)
            _augs.append(augs)
        _augs = np.concatenate(_augs, axis=0)
        Xs = torch.from_numpy(_augs.astype(np.float32)).to(device)
        hs = my_model(Xs)
        hs = torch.reshape(hs, [n_pics, n_samples, -1])

        loss = criterion(hs, mode="contrast", optimizer=optimizer)
        loss_log.append(loss.item())
        time_cost = (time.time()-start_time)/3600

        print('[Epoch %d][%d/%d]\tLoss: %.4f\tTime: %.4f hrs'
              % (i_epoch+1, i_round+1, n_round, loss.item(), time_cost))

        if (len(loss_log)+1)%20==0:
            curr_loss = np.mean(loss_log[-20:])
            print("------------------------")
            print("curr_loss", curr_loss, "best_loss", best_loss)
            print(model_path)
            if curr_loss<best_loss:
                best_loss = curr_loss
                torch.save(my_model.state_dict(), model_path)
                print(model_path, "Model Saved")
            else:
                print(model_path, "Model Not Saved")
            print("------------------------")
