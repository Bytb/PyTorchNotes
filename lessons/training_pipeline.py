# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#       - forward pass: compute prediction
#       - backward pass: gradients
#       - update weights

import torch
import numpy as np
import torch.nn as nn

#f = w * x
# f = 2 * x
x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32) #model only takes in tensors (have to be multiple dimensions: multiple features)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
x_pred = torch.tensor([5], dtype=torch.float32) #have to turn pred data into a tensor so that it can be passed into model
n_samples, n_features = x.shape #X.SHAPE = 4 x 1. This will set n_samples = 4 and n_features = 1
print(n_samples)
print(n_features)

input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

print(f'Prediction before training: f(5) = {model(x_pred).item():.3f}') #you can only pass tensors through model

#training
learning_rate = 0.01
n_iters = 50

loss = nn.MSELoss() #loss function
#optimizer function
#SGD = stochastic gradient descent (it takes in an array and the learning rate)
#NOTE: WHAT IS MODEL.PARAMETERS()?
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = model(x)

    #loss 
    l = loss(y, y_pred)

    #gradients = backwards pass
    l.backward() #does dl/dw

    #update weights
    optimizer.step()

    #zero gradients
    optimizer.zero_grad()

    if epoch % 5 == 0:
        #NOTE: WHAT IS 'b'? If it is the loss value, why not call it L?
        [w, b] = model.parameters() #b is the loss value
        print(f'epoch {epoch + 1}, w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prdiction after training: f(5) = {model(x_pred).item():.3f} ')