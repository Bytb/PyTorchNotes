import torch
import torch.nn as nn
import numpy as np

#cross entropy function
def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted)) #cross entropy function
    return loss 

# y must be one hot encoded
y = np.array([1, 0, 0])

# y_pred has probabilities 
y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(y, y_pred_good)
l2 = cross_entropy(y, y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')
print()


#---IN PYTORCH---
# NOTE: nn.CrossEntropyLoss applies: nn.LogSoft,ax + nn.NLLLoss (negative log likelihood loss)
# --> Do not Softmax the last layer!
# Y has class labels, not One-Hot
loss = nn.CrossEntropyLoss()

y = torch.tensor([2, 0, 1])
#size = number of samples * number of classes = 3 x 3
# classes refers to number of output options
# samples refers to how many predictions
y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]])

l1 = loss(y_pred_good, y)
l2 = loss(y_pred_bad, y)
print(l1.item())
print(l2.item())
print()

# _, will set the first part of the unpacked object to nothing
_, predictions1 = torch.max(y_pred_good, 1)
_, predictions2 = torch.max(y_pred_bad, 1)

print(predictions1)
print(predictions2)