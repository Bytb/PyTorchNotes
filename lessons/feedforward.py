# MNIST
# DataLoader, Trasnformation
# Multilayer Neural Net, activation function
# Loss and Optimizer
# Training loop (batch training)
# Model evaluation
# GPU Support
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader #we do not need to use Dataset because this is not a custom dataset

#device config (GPU Support)
# CUDA is NVIDIA's GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 784 #images are 28 x 28
hidden_size = 100 #number of nodes in NN
num_classes = 10 #digits 0 - 9
num_epochs = 2
batch_size = 100
learning_rate = 0.001

#import MNIST
#imports the data and stores it in root file './data
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

#this is the dataLoader that allows the data to be iterated over and used
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) #you do not need the test to be shuffled because it is for testing

#creating the NN class
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) #in this case, the output is the hidden_size
        self.relu = nn.ReLU() #activation function
        self.l2 = nn.Linear(hidden_size, num_classes) #output will be the num_classes
    
    #this is the construction of the NN
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

#creating NN model
model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer 
criterion = nn.CrossEntropyLoss() #you should not softmax the output because torch.CrossEntropy will take care of it
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #utilizing the Adam optimizer

#training loop
n_total_steps = len(train_loader) #this the number of iterations (remember: iterations * batch_size = epoch)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 100, 1, 28, 28
        # input_size = 784
        # NOTE: Ask JUSTIN about images and reshaping images (i think all this is doing is flattening the image)
        images = images.reshape(-1, 28*28).to(device) #sends to GPU 
        labels = labels.to(device) #sends to GPU

        #forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backwards
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1)%100 == 0:
            print(f'rpoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

#testing
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        #reshaping again to a 1D tensor
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        #getting predicted values
        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1) #essentially returning the highest probability of each senario
        n_samples += labels.shape[0] #number of samples
        n_correct += (predictions == labels).sum().item() #if the predictions equals the label then you sum it 
    
    acc = 100.0 * n_correct / n_samples
    print(f'Acc: {acc}')