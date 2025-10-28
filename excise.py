#实现一个Python类实现线性层 forward/backward
import numpy as np
class Linear:
    def __init_(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim)
        self.b = np.zeros(out_dim)
    
    def forward(self, x):
        self.x = x
        return x @self.W +self.b
    
    def backward(self, grad_output, lr=0.01):
        grad_W = self.x.T @grad_output
        grad_b = grad_output.sum(axis=0)
        self.W -= lr * grad_W
        self.b -= lr * grad_b

import numpy as np
class Linear():
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim)
        self.b = np.zeros(out_dim)

    def forward(self, x):
        self.x = x
        return x @self.W + self.b
    
    def backward(self, grad_output, lr=0.01):
        grad_W = self.x.T @grad_output
        grad_b = grad_output.sum(axis=0)
        self.W -= lr * grad_W
        self.b -= lr * grad_b

import numpy as np
class Linear():
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim)
        self.b = np.zeros(out_dim)

    def forward(self, x):
        self.x = x
        return x @self.W + self.b

    def backward(self, grad_output, lr=0.01):
        grad_W = self.x.T @grad_output
        grad_b = grad_output.sum(axis=0)
        self.W -= lr * grad_W
        self.b -= lr * grad_b

class Linear:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim)
        self.b = np.zeros(out_dim)

    def forward(self, x):
        self.x = x
        return x @self.W + self.b

    def backward(self, grad_output, lr= 0.01):
        grad_W = self.x.T @grad_output
        grad_b = grad_output.sum(axis=0)
        self.w -= lr * grad_W
        self.b -= lr *grad_b


#Create a Dataset
class MyDataset(Dataset):
    def _init_(self, base_df):
        x_df = base_df.copy()
        y_df = x_df.pop('label')
        x_df = x_df.values / 255 #Normalize values from 0 to 1
        x_df = x_df.reshape(-1, IMG_CHS, IMG_HEIGHT, IMG_WIDTH)
        self.xs = torch.tensor(x_df).folat().to(device)
        self.ys = torch.tensor(y_df).to(device)

    def _getitem_(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y
    
    def _len_(self):
        return len(self.xs)
    
class MyDataset(Dataset):
    def _init_(self, base_df):
        x_df = base_df.copy()
        y_df = x_df.pop('label')
        x_df = x_df.values / 255
        x_df = x_df.shape(-1, IMG_CHS, IMG_HEIGHT, IMG_WIDTH)
        self.xs = torch.tensor(x_df).float().to(device)
        self.ys = torch.tensor(y_df).to(device)
    
    def _getitem_(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y
    
    def _len_(self):
        return len(self.xs)




    
#数据清洗，从CSV中读入数据，去除缺失值并统计每列均值
import pandas as pd

df = pd.read_csv('data.csv')
df = df.dropna()
means = df.mean()


df = pd.read_csv('./data.csv')
df = pd.dropna()
means = df.mean()

#Softmax 函数
import numpy as np
def softmax(x):
    x = np.array(x)
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

import numpy as np # pip3 install numpy
def softmax(x):
    x = np.array(x)
    exp_x = np.exp(x- np.max(x)) # um numerische Stabilität zu gewährleisten (vermeiden overflow)
    return exp_x / np.sum(exp_x) # normalisierung (Wahrscheinlichkeitsverteilung)

def softmax(x):
    x = np.array(x)
    exp_x =np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)




#Logistic Regression
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, lr=0.1, epochs=1000):
    w = np.zeros(X.shape[1])
    for i in range(epochs):
        y_pred = sigmoid(np.dot(X, w))
        grad = np.dot(X.T, (y_pred - y)) / len(y)
        w -= lr * grad
    return w

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, lr=0.1, epochs=1000):
    w = np.zeros(X.shape[1])
    for i in range(epochs):
        y_pred = sigmoid(np.dot(X, w))
        grad = np.dot(X.T, (y_pred - y))/ len(y)
        w -= lr * grad
    return w

def logistic_regression(X, y, lr=0.1, epochs=1000):
    w = np.zeros(X.shape[1])
    for i in range(epochs):
        y_pred = sigmoid(np.dot(X, w))
        grad = np.dot(X.T, (y_pred - y )) / len(y)
        w -= lr * grad
    return w






#Linear
y_hat = wx + b

#ReLU
linear = wx + b
y_hat = linear * (linear > 0)

#Sigmoid
linear = wx + b
inf_to_zero = np.exp(-1 * linear)
y_hat = 1 / (1 + inf_to_zero)

#CrossEntropy
def cross_entropy(y_hat, y_actual):
    loss = log(y_hat) if y_actual else log(1 - y_hat)
    return -1 * loss

#FAST API 推理接口题
from fastapi import FastAPI, Body
import numpy as np

app = FastAPI
@app.post('/predict')
def predict(x: list = Body(...)):
    x = np.array
    y = np.sum(x)
    return {'result': float(y)}


#Stack (last in first out)
class stack():
    def __init__(self):
        self.data = []        #erstelle eine leere self.data, um die element des stacks zu speichern
    
    def push(self, x):
        self.data.append(x)   #fügt ein element oben auf den stapel,dazu list.append() verwendet

    def pop(self):
        if not self.data:
            return None
        return self.data.pop  #entfernt und gibt das oberste element zurück, wenn der stack leer ist. wird None zurückgegeben(keine fehlermeldung)
    
    def top(self):
        return self.data[-1] if self.data else None
    # gibt das oberste element zurück, ohne es zu lösen

class stack():
    def __init__(self):
        self.data = []

    def push(self, x):
        self.data.append(x)

        s.push(30)
        print(s.top())
        print(s.data)

        s = Stack()
        s.push(10)
        s.push(20)
        print(s.data)

    def pop(self):
        if not self.data:
            return None
        return self.data.pop
    
    def top(self):
        return self.data[-1] if self.data else None