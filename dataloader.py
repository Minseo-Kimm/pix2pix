import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from glob import glob
from skimage.transform import resize

class pix2pix_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, direction=0):
        self.data_dir = data_dir
        self.transform = transform
        self.direction = direction

        self.lst_data = sorted(list(glob(data_dir + '\*.jpg')))

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, idx):
        img = plt.imread(self.lst_data[idx])[:, :, :3]
        size = img.shape

        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        if img.dtype == np.uint8:
            img = img / 255.0

        if self.direction == 0:  # label: left, input: right
            data = {'label': img[:, :size[1]//2, :], 'input': img[:, size[1]//2:, :]}
        elif self.direction == 1:    # label: right, input: left
            data = {'label': img[:, size[1]//2:, :], 'input': img[:, :size[1]//2, :]}
        else:
            data = {'label': img}
        
        if self.transform: 
            data = self.transform(data)
        
        return data

# Transform 구현
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}
        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']
        label = (label - self.mean) / self.std
        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}
        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}
        return data

class RandomCrop(object):
    def __init__(self, shape):
        self.shape = shape
    
    def __call__(self, data):
        label, input = data['label'], data['input']
        h, w = data['label'].shape[:2]
        nh, nw = self.shape

        dh = np.random.randint(0, h - nh)
        dw = np.random.randint(0, w - nw)
        rh = np.arange(dh, dh + nh, 1)[:, np.newaxis]
        rw = np.arange(dw, dw + nw, 1)

        label = label[rh, rw]
        input = input[rh, rw]
        data = {'label': label, 'input': input}
        return data
        
class Resize(object):
    def __init__(self, shape):
        self.shape = shape
    
    def __call__(self, data):
        label, input = data['label'], data['input']
        label = resize(label, (self.shape[0], self.shape[1], self.shape[2]))
        input = resize(input, (self.shape[0], self.shape[1], self.shape[2]))

        data = {'label': label, 'input': input}
        return data