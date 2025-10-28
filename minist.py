import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.opti import Adam

#Visulization tools
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.transform.funtional as F
import matplotlib.pyplot as plt

#In PyTorch, use GPU in our operations by setting the device to cuda. The function torch.cuda.is_available() will confirm PyTorch can recognize the GPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.is_avaliable()

#Loading the Data into Memory, use Torchvision
train_set = torchvision.datasets.MNIST('./data/', train=True, download=True)
valid_set = torchvision.datasets.MNIST('./data/', train=False, download=True)

x_0, y_0 = train_set[0]
x_0
y_0
type(x_0)
y_0
type(y_0)

trans = transforms.Compose([transforms.ToTensor])
x_0_tensor = trans(x_0)

x_0_tensor.dtype
x_0_tensor.min()
x_0_tensor.max()

x_0_tensor.size()
x_0_tensor

x_0_tensor.device
x_0_gpu = x_0_tensor.cuda()
x_0_gpu.device

x_0_tensor.to(device).device

image = F.to_pil_image(x_0_tensor)
plt.imshow(image, cmap='gray')

trans = transforms.Compose([transforms.ToTensor()])
train_set.transform = trans
valid_set.transform = trans


batch_size = 32

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle = True)
valid_loader = DataLoader(valid_set, batch_size=batch_size)

#Creating the Model
layers = []
layers

#Flattening the image
test_matrix = torch_tensor(
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]
    ]
)
test_matrix

nn.Flatten()(test_matrix)

batch_test_matrix = test_matrix[None, :]
batch_test_matrix

nn.Flatten()(batch_test_matrix)

nn.Flatten()(test_matrix[:, None])

layers = [
    nn.Flatten()
]
layers

input_size = 1 * 28 * 28

layers = [
    nn.Flatten(),
    nn.Linear(input_size, 512),
    nn.ReLU(),
]
layers

n_classes = 10

layers = [
    nn.Flatten(),
    nn.Linear(input_size, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, n_classes)
]
layers

model = nn.Sequential(*layers)
model

model.to(device)

next(model.parameters()).device

model = torch.complie(model)
loss_function = nn.CrossentropyLoss()

optimizer = Adam(model.parameters)

train_N = len(train_loader.dataset)
valid_N = len(valid_loader.dataset)

#Function to calculate the accuracy for each batch
def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.views_as(pred)).sum.item()
    return correct / N

#Train function
def train():
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
        print('Train - Loss: {:.4f} Accuracy:{:.4f}'.format(loss, accuracy))

#Validate function
def validate():
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    print('Valid - loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

#Training loop
epochs = 5

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train()
    validate()

prediction = model(x_0_gpu)
prediction

prediction.argmx(dim=1, keepdim=True)

y_0

#Clean the memory
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
