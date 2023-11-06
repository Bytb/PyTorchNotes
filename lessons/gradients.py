import torch

x = torch.randn(3, requires_grad=True) #tells pytorch to store gradients
y = x + 2 #since the operation is (+), the back propogation fn is called 'AddBackward'
print(y)

z = y * y * 2
print(z) #this will give a fn = MulBackwards

z = z.mean()
print(z) #fn = MeanBackwards

#NOTE:
#QUESTION: how does back propogation work with gradients?
#QUESTION: why is the value stored in x?
z.backward() # dz/dx NOTE:this can only be used on scalar values (1 value) (z.mean() converts z from a vector to scalar)
print(x.grad)

#How to make the tensor not require a grad function
#1. x.requires_grad_(False)
x.requires_grad_(False) #the _ after the function means that it will update the value without having to do x = operation
#2. x.detach()
y = x.detach() #create a new tensor with the same values
#3. with torch.no_grad()
with torch.no_grad():
    y = x + 2