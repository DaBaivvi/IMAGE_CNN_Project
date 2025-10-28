#Convolutional neural network
import torch.nn as nn
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda' if torch.cuda_is_avaliable() else 'cpu')
torch.cuda.is_avaliable()

#Loading and preparing the Data
train_df = pd.read_csv('data/als_data/sign_mnist_train.csv')
valid_df = pd.read_csv('data/als_data/sign_mnist_valid.csv')

#ASL data is already flattened
sample_df = train_df.head().copy() #Grap the top 5 raws
sample_df.pop('label')
sample_x = sampel_df.values
sample_x

sample_x.shape

IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHS = 1

sample_x = sample_x.resahpe(-1, IMG_CHS, IMG_HEIGHT; IMG_WIDTH)
sample_x.sahpe

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

#Create a DataLoader
BATCH_SIZE = 32

train_data = MyDataset(train_df)
train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
train_N = len(train_loader.dataset)

valid_data = MyDataset(valid_df)
valid_loader = DataLoader(valid_data, batch_size = BATCH_SIZE)
valid_N = len(valid_loader.dataset)

batch = next(iter(train_loader))
batch

batch[0].shape
batch[1].shape

#Create a convolutional Model
n_classes = 24
kernel_size = 3
flattened_img_size = 75 * 3 * 3

model = nn.Sequential(
    #first convolution
    nn.Conv2d(IMG_CHS, 25, kernel_size, stride=1, padding=1),
    nn.BatchNorm2d(25),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),
    
    #second convolution
    nn.Conv2d(25, 50, kernel_size, stride=1, padding=1),
    nn.BatchNorm2d(50),
    nn.ReLU()
    nn.MaxPool2d(2, stride=2),

    #third convolution
    nn.Conv2d(50, 75, kernel_size, stride=1, padding=1),
    nn.BatchNorm2d(75),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2)

    #Flatten to dense
    nn.Flatten(),
    nn.Linear(flattened_img_size, 512),
    nn.Dropout,
    nn.ReLU,
    nn.Linear(512, n_classes)
)

#Summarizing the mdoel
model = torch.compile(model.to(device))
model

loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.views_as(pred).sum().item())
    return correct / N

#Training the model
def validate():
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    print('Train - Loss: {:.4f} Accuracy {:.4f}'.format(loss, accuracy))

def train():
    loss = 0
    accuracy = 0

    model.train()
    with torch.no_grad():
        for x, y in valid_loader:
            output = model(x)
            optimizer.zero_grad()
            batch_loss = loss_function(output, y)
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
            accuracy += get_batch_accuracy(output, y, train_N)
    print('Train - Loss:{:.4f} Accuracy:{:.4f}'.format(loss, accuracy))


epochs = 20
for i in range(epochs):
    print('Epoch: {}'.format(epoch))
    train()
    validate()

#Clean the Memory
import IPython
app =IPython.Application.instance()
app.kernel.do_shutdown(True)