import torch
import numpy as np

#creates an empty tensor and can specify dimensions
#dtype will specify the type that will be in tensor 
x = torch.empty(2, 2, 1, dtype=torch.float16)

#basic math operations
a = torch.rand(2, 2)
b = torch.rand(2, 2)
#add
c = a + b
c = torch.add(a, b)
#sub
c = a - b
c = torch.sub(a, b)
#multiply
c = a * b
c = torch.mul(a, b)
#divide
c = a / b
c = torch.div(a, b)

#slicing
a = torch.rand(5,3) #creating a large tensor (5 x 3)
b = a[1, :] #getting all the first elements of each row
c = a[2,1].item() #single elements ('.item' will return the actual value that is in the tensor but can only be used if the tensor contains one value)

#reshaping the tensor
x = torch.rand(4,4)
y = x.view(16) #turns it in 1D
z = x.view(-1, 8) #will assume second value based off other value


#NOTE: there are a lot of errors when it comes to Numpy not working on a tensor GPU (ask Justin about this)
#converting from numpy to torch tensor
a = torch.ones(5)
b = a.numpy() #converts to numpy array

c = np.ones(5)
d = torch.from_numpy(c)