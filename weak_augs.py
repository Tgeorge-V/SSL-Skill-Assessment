import numpy as np
import scipy.stats as stats
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import random
import collections

import torch
from torchvision import transforms

def img_randomaffine(img):
    degree=random.randint(1,5)
    translate = random.randint(0,5)/100
    trans = transforms.RandomAffine(degrees=degree, translate=[translate,translate])
    result = trans(img)
    return result

def img_randomhorizontalhlip(img):
    p=0.1
    trans = transforms.RandomHorizontalFlip(p=p)
    result = trans(img)
    return result

def img_randomverticalflip(img):
    p=0.1
    trans = transforms.RandomVerticalFlip(p=p)
    result = trans(img)
    return result

def img_randomrotation(img):
    degree=random.randint(1,5)
    trans = transforms.RandomRotation(degrees=degree)
    result = trans(img)
    return result

def get_weak_augments_list():
    l = [
        (img_randomaffine),
        (img_randomhorizontalhlip),
        (img_randomverticalflip),
        (img_randomrotation)
    ]
    return l

class weak_img_aug:
    def __init__(self, num_augs):
        self.augment_list = get_weak_augments_list()
        assert 1<=num_augs<=len(self.augment_list)
        self.n=num_augs
    def __call__(self, img):
        max_num = np.random.randint(1, self.n + 1)
        ops = random.choices(self.augment_list, k=max_num)
        transform = []
        for op in ops:
            img = op(img)
        result = transforms.ToTensor()(img)
        return result