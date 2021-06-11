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

from lib.utils import stl10
from lib.utils_torch import Identity, Loss, Net34
from lib.utils import normalize, read_all_imgs, extractor, plot_loss

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment

result = stl10("./data/stl10_binary") # set the data path

print(result["unlabeled_X"].shape,
      result["train_X"].shape,
      result["train_y"].shape,
      result["test_X"].shape,
      result["test_y"].shape)

lr = 5e-4
gpus = [0,1] # use multiple gpus
n_pics = 40*len(gpus)
n_samples = 4
model_path = "./models/res34_model_training_with_STL_"+str(time.time()).replace(".","")+".pth"

if len(gpus)>0:
    device = torch.device("cuda:"+str(gpus[0]))
else:
    device = torch.device("cpu")

mean = np.mean(result["unlabeled_X"]/255.)
std = np.std(result["unlabeled_X"]/255.)

etr = extractor()

my_model = Net34().to(device)
if len(gpus)>1: my_model = nn.DataParallel(my_model, gpus)

optimizer = optim.Adam(my_model.parameters(), lr)
criterion = Loss(device)

best_loss = 0.0
i_epoch = -1
loss_log = []
start_time = time.time()

while True:
    i_epoch+=1
    _indices = np.random.choice(len(result["unlabeled_X"]), len(result["unlabeled_X"]), replace=False)
    n_round = int(len(result["unlabeled_X"])/n_pics)
    for i_round in range(n_round):
        _augs = []
        indices = _indices[i_round * n_pics : (i_round+1) * n_pics]
        for index in indices:
            img = result["unlabeled_X"][index]
            augs = etr.extract(img, n_augs=n_samples, target_size=224, resolution_aug="False")
            augs = (augs.transpose([0,3,1,2])-mean)/std
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

        if len(loss_log)%5==0:
            curr_loss = np.mean(loss_log[-5:])
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
