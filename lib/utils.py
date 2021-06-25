import os
import cv2
import time
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models

from imgaug import augmenters as iaa

def plot_loss(loss_log):
    data = np.log10(loss_log)
    gap = max(data)-min(data)
    plt.figure(figsize=(10, 6))
    plt.title('Loss history (log10)')
    plt.plot(data, '.', alpha=0.1)
    for _g in [0,1,3,7,15]:
        y_value = min(data)+_g*gap/16
        plt.plot([0,len(data)], [y_value, y_value], 'r', linewidth=1, alpha=0.2)
        plt.text(0, y_value, str(np.round(y_value, 3)),
                 fontdict=dict(color='r', fontsize=11),)
    plt.show()

def normalize(Xs, mean, std):
    Xs_ = np.copy(Xs)
    for i in range(Xs_.shape[1]):
        Xs_[:,i,...] = (Xs_[:,i,...]-mean[i])/std[i]
    return Xs_

def cifar10(path):

    def load_CIFAR_batch(filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='iso-8859-1')
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000,3072)
            Y = np.array(Y)
            return X, Y

    def load_CIFAR10(ROOT):
        """ load all of cifar """
        xs = []
        ys = []
        for b in range(1,6):
            f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
            X, Y = load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)
        Xtr = np.concatenate(xs)
        Ytr = np.concatenate(ys)
        del X, Y
        Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
        return Xtr, Ytr, Xte, Yte

    cifar10_dir = '../input/cifar-10-batches-py/'
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    X_train, y_train, X_test, y_test = load_CIFAR10(path)
    
    trains = {}
    tests = {}
    trains['data'] = X_train
    trains['labels'] = y_train
    trains['label_names'] = classes
    tests['data'] = X_test
    tests['labels'] = y_test
    tests['label_names'] = classes
    
    return trains, tests

def cifar100(path):
    files = ['train', 'test']
    
    coarse_names = ['aquatic mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
                    'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
                    'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
                    'large_omnivores_and_herbivores', 'medium-sized_mammals', 'non-insect_invertebrates',
                    'people', 'reptiles', 'small_mammals', 'trees', 'vehicles_1', 'vehicles_2']

    fine_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
                  'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
                  'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
                  'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
                  'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
                  'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                  'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                  'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                  'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                  'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                  'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                  'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                  'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                  'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                  'worm']
    
    with open(os.path.join(path, files[0]), 'rb') as fo:
        trains = pickle.load(fo, encoding='iso-8859-1')
            
    with open(os.path.join(path, files[1]), 'rb') as fo:
        tests = pickle.load(fo, encoding='iso-8859-1')
            
    trains['coarse_names'] = coarse_names
    trains['fine_names'] = fine_names
    tests['coarse_names'] = coarse_names
    tests['fine_names'] = fine_names

    return trains, tests

def stl10(path):
    def read_labels(path_to_labels):
        with open(path_to_labels, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8)
            return labels

    def read_all_images(path_to_data):
        with open(path_to_data, 'rb') as f:
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 3, 2, 1))
            return images
    
    names = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
    path_ux = os.path.join(path, "unlabeled_X.bin")
    path_trx = os.path.join(path, "train_X.bin")
    path_try = os.path.join(path, "train_y.bin")
    path_tex = os.path.join(path, "test_X.bin")
    path_tey = os.path.join(path, "test_y.bin")
    ux = read_all_images(path_ux)
    X = read_all_images(path_trx)
    _X = read_all_images(path_tex)
    y = read_labels(path_try)
    _y = read_labels(path_tey)
    result = {"unlabeled_X": ux, "train_X": X, "train_y": y, "test_X": _X, "test_y": _y, "class_names": names}
    return result

def read_all_imgs(path, suffix=[".jpg", ".jpeg", ".png"], verbose=0, maximum=0, _iter=True):
    """
    verbose: gap
    """
    def helper(path, suffix, verbose=(0,0), maximum=0):
        """
        verbose: start, gap
        """
        result = []
        for item in os.listdir(path):
            p = os.path.join(path, item)
            if os.path.isdir(p):
                result = result+helper(p, suffix, verbose=(len(result)+verbose[0], verbose[1]))
            elif any([s==item[-len(s):].lower() for s in suffix]):
                result.append(p)
                if verbose[1]>0 and (len(result)+verbose[0])%verbose[1]==0:
                    print(len(result)+verbose[0])

            if maximum>0 and (len(result)+verbose[0])>=maximum:
                return result

        return result
    
    if _iter:
        suffix_ = [x.lower() for x in suffix]
        result = helper(path, suffix_, (0, verbose), maximum)
        if verbose>0 and len(result)%verbose!=0:
            print(len(result))
        return result
    else:
        result = [os.path.join(path, item) for item in os.listdir(path)]
        result = [x for x in result if any([s==x[-len(s):].lower() for s in suffix])]
        return result

class extractor:
    
    def __init__(self, aug_seq=None):
        if aug_seq is None:
            self.aug_seq = iaa.Sequential([
                iaa.Affine(rotate=(-35, 35)),
                # iaa.AdditiveGaussianNoise(scale=(0, 20)),
                iaa.AddToHue((-25, 25)),
                iaa.AddToSaturation((-150, 50)),
                iaa.MultiplyAndAddToBrightness(mul=(0.75, 1.25), add=(-50, 50)),
                # iaa.Cutout(nb_iterations=(0, 2), size=0.15),
                # iaa.Crop(percent=(0, 0.3)),
            ])
        else:
            self.aug_seq = aug_seq

    def resize_short(self, img, size):
        short = min(img.shape[:2])
        scale = size/short
        new_shape = (int(max(img.shape[1] * scale, size)), int(max(img.shape[0] * scale, size)))
        resized = cv2.resize(img, new_shape)
        return resized

    def rand_crop(self, img, size):
        """
        size = (w, h)
        """
        assert img.shape[1]>=size[0] and img.shape[0]>=size[1]
        start_w = np.random.randint(1+img.shape[1]-size[0])
        start_h = np.random.randint(1+img.shape[0]-size[1])
        return np.copy(img[start_h:(start_h+size[1]), start_w:(start_w+size[0])])
        
    def extract(self, img, n_augs, target_size=224, min_crop_rate=0.2, fix_crop_rate=False,
                aug_seq=None, resolution_aug="True", isDivide255=True):
        img_ = self.resize_short(img, target_size*2)
        min_size = target_size*2*min_crop_rate
        
        imgs = []
        for _ in range(n_augs):
            if aug_seq is None:
                if fix_crop_rate:
                    size = int(min_size)
                else:
                    size = int(min_size + (target_size*2-min_size)*np.random.random())
                cropped = self.rand_crop(img_, (size, size))
                if np.random.random()<0.5:
                    cropped = cv2.flip(cropped, 1)
            else:
                if fix_crop_rate:
                    size = int(min_size)
                else:
                    size = int(min_size + (target_size*2-min_size)*np.random.random())
                cropped = self.rand_crop(img_, (size, size))
            if resolution_aug and np.random.random()<0.7:
                size = int(target_size * (np.random.random()*0.8+0.2))
                cropped = cv2.resize(cropped, (size, size))
            cropped = cv2.resize(cropped, (target_size, target_size))
            imgs.append(cropped)
        if aug_seq is None:
            result = self.aug_seq(images=np.array(imgs))
        else:
            result = aug_seq(images=np.array(imgs))
        if isDivide255:
            result = result/255.
        return result
    
    def extract_middle(self, img, target_size=224, isDivide255=True):
        img_ = self.resize_short(img, target_size)
        if img_.shape[0]>img_.shape[1]:
            cut_off = (img_.shape[0]-img_.shape[1])//2
            img_ = img_[cut_off:(cut_off+img_.shape[1]), :]
        elif img_.shape[0]<img_.shape[1]:
            cut_off = (img_.shape[1]-img_.shape[0])//2
            img_ = img_[:, cut_off:(cut_off+img_.shape[0])]
        if isDivide255:
            img_ = img_/255.
        return np.array([img_])
