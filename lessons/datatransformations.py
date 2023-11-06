import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
#complete list of built in transformations:
# https://pytorch.org/vision/0.9/transforms.html

#---IMAGE DATASETS---
# CenterCrop, Grayscale, Pad, RandomAffine, Random Crop,
# RandomHorizontalFlip, RandomRotation, Resize, Scale

#---TENSORS---
# LinearTransformation, Normalize, RandomErasing

#---CONVERSION---
# toPILImage: from tensor to ndrarray
# ToTensor: from numpy.ndarray or PILImage

#---GENERIC---
# Use Lambda

#---USE MULTIPLE---
# composed = transforms.Compose([TRANSFORMATIONS])

#Notice how this has Dataset as the parent class
class WineDataset(Dataset):
    def __init__(self, transform=None) -> None:
        xy = np.loadtxt('/Users/calebfernandes/Desktop/LearningPyTorch/lessons/data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = (xy[:, 1:])
        self.y = (xy[:, [0]])
        self.n_samples = xy.shape[0]
        #new transform variable
        self.transform = transform

    def __getitem__(self, index):
        #dataset[0]
        sample = self.x[index], self.y[index]

        #applies the transformation variable to the sample if it exists
        if self.transform:
            sample=self.transform(sample)
        
        return sample

    def __len__(self):
        #len(dataset)
        return self.n_samples

#NOTE: HOW DOES __call__ work??
#I think it calls itself as a function as soon as it is made
class ToTensor:
    def __call__(self, sample):
        inputs, target = sample
        return torch.from_numpy(inputs), torch.from_numpy(target)
    
class MulTransform:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

dataset = WineDataset(transform=ToTensor())

first_data = dataset[0]
features, targets = first_data
print(type(features), type(targets))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)

first_data = dataset[0]
features, targets = first_data
print(features, targets)
print(type(features), type(targets))