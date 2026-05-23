import torch
import torch.nn as nn
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

from timeit import default_timer as timer

train_data = datasets.MNIST(root='data',train=True,download=True,transform=ToTensor(),target_transform=None)
test_data = datasets.MNIST(root='data',train=False,download=True,transform=ToTensor())

device= 'cpu'

img,label = train_data[0]
print(img)
print(label)

print(img.shape)
plt.imshow(img.squeeze())

print(len(train_data.data),len(train_data.targets),len(test_data.data),len(test_data.targets))

class_names = train_data.classes
print(class_names)

BATCH_SIZE = 32
train_dataloader = DataLoader(train_data,BATCH_SIZE,shuffle=True)
test_dataloader = DataLoader(test_data,BATCH_SIZE,shuffle=False)

print(len(train_dataloader))

print(len(test_dataloader))

class MNISTModel(nn.Module):
  def __init__(self,input_shape,hidden_units,output_shape):
    super().__init__()
    self.block1 = nn.Sequential(
                  nn.Conv2d(in_channels=input_shape,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
                  nn.ReLU(),
                  nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
                  nn.ReLU(),
                  nn.MaxPool2d(kernel_size=2,stride=2))
    self.block2 = nn.Sequential(
              nn.Conv2d(hidden_units,hidden_units,kernel_size=3,padding=1),
              nn.ReLU(),
              nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,padding=1),
              nn.ReLU(),
              nn.MaxPool2d(kernel_size=2,stride=2))
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*7*7,out_features=output_shape)
    )

  def forward(self,x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.classifier(x)
    return x

torch.manual_seed(42)
model = MNISTModel(1,12,len(class_names)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),lr=0.1)

def accuracy_fn(y_pred,y_test):
  correct = torch.eq(y_pred,y_test).sum().item()
  return (correct*100)/len(y_test)

def train_step(model,data_loader,loss_fn,optimizer,accuracy_fn,device):
  train_loss,train_acc = 0,0
  model.to(device)
  model.train()
  for X,y in data_loader:
    X,y = X.to(device),y.to(device)
    y_logits = model(X)
    y_preds = y_logits.argmax(dim=1)
    loss = loss_fn(y_logits,y)
    train_loss += loss
    train_acc += accuracy_fn(y_preds,y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

  train_loss /= len(data_loader)
  train_acc /= len(data_loader)
  print(f'Train loss: {train_loss} Train accuracy: {train_acc}')

def test_step(model,data_loader,loss_fn,accuracy_fn,device):
  test_loss,test_acc = 0,0
  model.to(device)
  model.eval()
  with torch.inference_mode():
    for X,y in data_loader:
      X,y = X.to(device),y.to(device)
      y_logits = model(X)
      test_loss += loss_fn(y_logits,y)
      y_preds = y_logits.argmax(dim=1)
      test_acc += accuracy_fn(y_preds,y)

    test_loss /= len(data_loader)
    test_acc /= len(data_loader)

  print(f"Test loss: {test_loss} Test acc: {test_acc}")

def eval_model(model,data_loader,loss_fn,accuracy_fn,device):
  test_loss,test_acc = 0,0
  model.eval()
  with torch.inference_mode():
    for X,y in data_loader:
      X,y = X.to(device),y.to(device)
      y_logits = model(X)
      y_preds = y_logits.argmax(dim=1)
      test_loss += loss_fn(y_logits,y)
      test_acc += accuracy_fn(y_preds,y)
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)

  print(f"Loss: {test_loss} Accuracy: {test_acc}")

def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

train_time_start_model = timer()
torch.manual_seed(42)
epochs = 3
for epoch in range(epochs):
  print(f"Epoch:{epoch}")
  train_step(model,train_dataloader,loss_fn,optimizer,accuracy_fn,device)
  test_step(model,test_dataloader,loss_fn,accuracy_fn,device)
train_time_end_model = timer()
total_train_time_model = print_train_time(start=train_time_start_model,
                                           end=train_time_end_model,
                                           device=device)

eval_model(model,test_dataloader,loss_fn,accuracy_fn,device)

y_preds = []
with torch.inference_mode():
  for X,y in test_dataloader:
    X,y = X.to(device),y.to(device)
    y_logits = model(X)
    y_pred = torch.softmax(y_logits,dim=1).argmax(dim=1)
    y_preds.append(y_pred.cpu())
y_preds_tensor = torch.cat(y_preds)


confmat = ConfusionMatrix(num_classes = len(class_names),task='multiclass')
confmat_tensor = confmat(preds=y_preds_tensor,target=test_data.targets)

fig,ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),class_names=class_names,figsize=(10,7))