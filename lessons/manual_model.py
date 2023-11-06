import numpy as np

#f = w * x


# f = 2 * x
x = np.array([1, 2, 3, 4], dtype=np.float32)
y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.0

#calculation model prediction
def forward(x):
    return w * x

#calculate loss
#loss = MSE (y_pred - y)^2
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

#calculate gradient
#MSE = 1/N * (w*x - y)**2
# dFunction/dw = (1/N) * (w*x - y) * 2x
def gradient(x, y, y_pred):
    return np.dot(2*x, y_pred - y).mean()

print(f'Prdiction before training: f(5) = {forward(5):.3f} ')

#training
learning_rate = 0.01
n_iters = 20
for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = forward(x)

    #loss 
    l = loss(y, y_pred)

    #gradients
    dw = gradient(x, y, y_pred)

    #update weights
    w -= learning_rate * dw

    if epoch % 2 == 0:
        print(f'epoch {epoch + 1}, w = {w:.3f}, loss = {l:.8f}')

print(f'Prdiction after training: f(5) = {forward(5):.3f} ')