# -*- coding: utf-8 -*-
"""
CIFAR100_Pretrained

"""

import torch
import torch.nn as nn
from torch.utils import model_zoo
import torchvision
import torchvision.transforms as transforms

#Initializing parameters:
batch_size = 100
learn_rate = 0.001
scheduler_step_size = 5
scheduler_gamma = 0.5
num_epochs = 50

#Setting up transforms:
transform_train = transforms.Compose([transforms.RandomRotation(10),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()
                                     ])

#Loading data:
train_dataset = torchvision.datasets.CIFAR100(root = '~/scractch/', train=True, transform=transform_train, download=True)
test_dataset = torchvision.datasets.CIFAR100(root = '~/scractch/', train=False, transform=transform_train, download=True)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = 8)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False, num_workers = 8)

#Defining upsampling of the image so it can be inputted to the pretrained resnet18 model:
def upsample(x):
  up = nn.Upsample(scale_factor=7, mode='bilinear')
  return(up(x))

#Defining a function to load the resnet18 model:
def resnet18(pretrained=True, progress=True):
  model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 4, 4, 2])

  if pretrained:
    model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth', model_dir='./'), strict = False)
  return(model)

#Initializing GPU:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model = resnet18(pretrained = True).to(device)
 
#Initializing loss function, optimizer and scheduler:   
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), 
                                lr = learn_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                            step_size = scheduler_step_size, 
                                            gamma = scheduler_gamma)

#Initializing lists to store train and test accuracies:
train_acc_list = []
test_acc_list = []

#Running the train and test iterations for the model:
for epochs in range(num_epochs):
    scheduler.step()
    correct = 0
    total = 0
    print('Current epoch: \t\t', epochs+1, '/', num_epochs)
    
    #Training:
    for images, labels in train_loader:
        #images = images.reshape(-1, 16*16)
        images = images
        images = upsample(images)
        images = images.to(device)
        labels = labels.to(device)
        
        #Forward propagation:
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total = total + labels.size(0)
        correct = correct + (predicted == labels).sum().item()
        
        #Backward propagation:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_acc = correct/total
    print('Training accuracy: \t', train_acc)
    train_acc_list.append(train_acc)
    
    #Testing:
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images
            images = upsample(images)
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()
    test_acc = correct/total
    print('Test Accuracy: \t\t', test_acc)
    print('**************************************************')
    test_acc_list.append(test_acc)
    model.train()
    
#print(train_acc_list)
#print(test_acc_list)
