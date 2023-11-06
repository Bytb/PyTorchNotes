#
# epoch = 1 forward and backward pass of ALL training samples
# batch_size = number of training samples in one forward and backward pass
# number of iterations = number of passes, each pass using [batch_size] number of samples
# ex. 100 samples, batch_size = 20 --> 100/20 = 5 iterations for 1 epoch
#
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

#Notice how this has Dataset as the parent class
class WineDataset(Dataset):
    def __init__(self) -> None:
        #data loading
        #loading in dataset
        xy = np.loadtxt('/Users/calebfernandes/Desktop/LearningPyTorch/lessons/data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        #splitting the values into the dependent and independent variables
        #NOTE: look up the notation for SPLICING arrays 
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        #dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        #len(dataset)
        return self.n_samples

dataset = WineDataset()
#DataLoader allows you to iterate over the dataset (move through the samples and features, you also setup the batchsize and shuffle property in here)
dataLoader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

#training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4) #4 is because the batch size is 4
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    #i is the row that you are in
    #inputs and labels is unpacked by the enumerate()
    #inputs are the independent variables
    #labels are the dependent variables associated with each sample
    for i, (inputs, labels) in enumerate(dataLoader): #remember that dataLoader is an interable version of the dataset
        # forward backward , update
        if (i + 1) % 5 == 0:
            print(f'epoch: {epoch+1}/{num_epochs}, step: {i+1}/{n_iterations}, inputs: {inputs.shape}')