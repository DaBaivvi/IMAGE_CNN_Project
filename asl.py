import torch.nn as nn
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.is_available()

#Loading the data (Panda)
train_df = pd.read_csv('data/asl_data/sign_mnist_train.csv')
valid_df = pd.read_csv('data/asl_data/sign_mnist_vaild.csv')

#Exploring the Data
train_df.head()

#Extracting the labels
y_train = train_df.pop('label')
y_valid = valid_df.pop('label')
y_train

#Extracting the Images
x_train = train_df.values
x_valid = valid_df.values
x_train

#Summarizing the training and validation data
x_train.shape
y_train.shape

x_valid.shape
y_valid.shape

#Visualizing the Data
import matplotlib.pyplot as plt
plt.figure(figsize=(40, 40))

num_images = 20
for i in range(num_images):
    row = x_train[i]
    label = y_train[i]

    image = row.reshape(20, 20)
    plt.subplot(1, num_images, i+1)
    plt.title(label, fontdict={'fontsize': 30})
    plt.axis('off')
    plt.imshow(image, camp='gray')

#Normalize the image data
x_train.min()
y_train.max()

x_train = train_df.values / 255
x_valid = valid_df.values / 255

#Custom Datasets
class MyDataset(Dataset):
    def _init_(self, x_df, y_df):
        self.xs = torch.tensor(x_df).float().to(device)
        self.ys = torch.tensor(y_df).to(device)

        #return images and labels
        def _getitem_(self, idx):
            x = self.xs[idx]
            y = self.ys[idx]
            return x, y
        
        def _len_(self):
            return len(self.xs)

#DataLoader for model Training
BATCH_SIZE = 32
train_data = MyDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size = BATCH_SIZE)
train_N = len(train_loader.dataset)

valid_data = MyDataset(x_valid, y_valid)
valid_loader = DataLoader(valid_data, batch_size = BATCH_SIZE)
valid_N = len(valid_data.dataset)

train_loader

#Iterable
batch = next(iter(train_loader))
batch

#Batch has 2 values (x, y). The first dimension of each should have 32 values, which is the batch_size
batch[0].shape
batch[1].shape

#Build the model (sequential model)
input_size = 28 * 28
n_classes = 24

model = nn.Sequential(
    nn.Flaten(),
    nn.Linear(input_size, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, n_classes)
)
#Combine compiling the model and sending it to the GPU in one step
model = torch.complie(model.to(device))
model

loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

#Training the model
def train():
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        output = model(x)
        optimizer.zero.grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
        print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

#Calculating the Accuracy
def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N

#The Training Loop
epochs = 20
for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train()
    validate()

#Clean the memory
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)        
