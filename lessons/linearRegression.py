# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#       - forward pass: compute prediction
#       - backward pass: gradients
#       - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#0) prepare data
#x_numpy is a numpy array that is of size/ dimensions [n_samples x n_features]
#y_numpy is a singular array that contains all of the values [n_samples]
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

#converts to torch frmo numpy array and the datatype from a double to a numpy float
x = torch.from_numpy(x_numpy.astype(np.float32)) 
y = torch.from_numpy(y_numpy.astype(np.float32))
#you need to reshape the data because right now it is a 1D array but it needs to be a 2D tensor
y = y.view(y.shape[0], 1) #view will reshape data so it is a 100 x 1 instead of a 100 (x is already 2D so it does not need to be changed)

n_samples, n_features = x.shape #sample is the same thing as saying row

#1) model
input_size = n_features #how many dimensions/features are in input
output_size = 1 #how many things are ouputed
model = nn.Linear(input_size, output_size) #Linear takes in the number of features considered and the number that should be outputted

#2) loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    #forward pass and loss
    y_predicted = model(x) #forward pass by sending the x training data through to the model (x = training data)
    loss = criterion(y_predicted, y) #loss function calculated using the y_predicted and the actual values (y = test values)

    #backward pass 
    loss.backward() #backward pass on the loss function to find the gradient in respect of dLoss / dx

    #update
    optimizer.step() #this is the step size in the direction of the loss function

    #zero out the weights before the next iteration
    optimizer.zero_grad()

    if(epoch+1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss: {loss.item():.4f}')

#plot
predicted = model(x).detach() #this makes it so that this model is not recorded in the TensorBoard graph
plt.plot(x_numpy, y_numpy, 'ro')
plt.plot(x_numpy, predicted, 'b')
plt.show()