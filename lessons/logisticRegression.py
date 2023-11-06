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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) prepare data
bc = datasets.load_breast_cancer() #loading in dataset 
# x is the list of all the independent variables
# y is the list of all the corresponding dependent variables
x, y = bc.data, bc.target

n_samples, n_features = x.shape

# x_train = all the independent features used to train the model
# x_test = all the independent features used to test the model
# y_train = the dependent output used for training
# y_test =  the dependent output used for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

#scale
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#converting to tensors
x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

#reshaping the y because they are in the shape of [:0] instead of [:1]
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


#ASK JAKE WHAT THIS IS
# 1) model
#this is important for when creating logistic models that have multiple layers
#it references the nn.Module superclass
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        #super() references the superclass and the point of this is to be able to import the modules from the superclass
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    

model = LogisticRegression(n_features)

# 2) loss and optimizer
learning_rate = 0.01
BCE = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    #forward pass
    y_pred = model(x_train)
    loss = BCE(y_pred, y_train)

    #backward pass
    loss.backward()

    #updates
    optimizer.step()

    #empty grad
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss: {loss.item():.4f}')

with torch.no_grad():
    y_pred = model(x_test)
    y_predicted_cls = y_pred.round()
    #.eq() will return true each time that predicted == y_test and then .sum() will sum them all and / y_test.shape will divide by total number 
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy: {acc:.4f}')