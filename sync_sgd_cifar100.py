# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:46:08 2019

@author: Pulkit Dixit
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.autograd import Variable

import os
import subprocess
from mpi4py import MPI

#Setting up the nodes for distributed training:
cmd = "/sbin/ifconfig"
out, err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
    stderr=subprocess.PIPE).communicate()
ip = str(out).split("inet addr:")[1].split()[0]

name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_nodes = int(comm.Get_size())

ip = comm.gather(ip)

if rank != 0:
  ip = None

ip = comm.bcast(ip, root=0)

os.environ['MASTER_ADDR'] = ip[0]
os.environ['MASTER_PORT'] = '2222'

backend = 'mpi'
dist.init_process_group(backend, rank=rank, world_size=num_nodes)

dtype = torch.FloatTensor

print('Rank: ', rank)
print('Number of nodes: ', num_nodes)

#-------------------------------------------

#Initializing parameters:
batch_size = 100
learn_rate = 0.001
scheduler_step_size = 13
scheduler_gamma = 0.1
num_epochs = 50

#Creating transforms:
transform_train = transforms.Compose([transforms.RandomRotation(10),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()
                                     ])

#Loading data:
train_dataset = torchvision.datasets.CIFAR100(root = '~/scractch', train=True, transform=transform_train, download=False)
test_dataset = torchvision.datasets.CIFAR100(root = '~/scractch', train=False, transform=transform_train, download=False)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False, num_workers = 0)

#Defining a 3x3 convolution:
def conv3x3(in_channels, out_channels, stride = 1):
  return nn.Conv2d(in_channels, out_channels, stride = stride, kernel_size = 3, padding = 1)

#Defining the basic block:
class BasicBlock(nn.Module):
  #expansion = 1
  def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(in_channels, out_channels, stride)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace = True)
    self.conv2 = conv3x3(out_channels, out_channels)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.downsample = downsample
    self.stride = stride
    
  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    if self.downsample is not None:
      residual = self.downsample(x)
    out += residual
    out = self.relu(out)
    return out

class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes = 100):
    super(ResNet, self).__init__()
    
    #Pre-basic block convolution:
    self.in_channels = 32
    self.conv = conv3x3(3, self.in_channels)
    self.bn = nn.BatchNorm2d(self.in_channels)
    self.relu = nn.ReLU(inplace = True)
    self.drop_out = nn.Dropout(0.1)
    
    #Basic block 1:
    self.layer1 = self.make_layer(block, 32, layers[0])
    
    #Basic block 2:
    self.layer2 = self.make_layer(block, 64, layers[1], 2)
    
    #Basic block 3:
    self.layer3 = self.make_layer(block, 128, layers[2], 2)
    
    #Basic block 3:
    self.layer4 = self.make_layer(block, 256, layers[3], 2)
    
    #Post Block pooling and linearization:
    self.maxpool = nn.AdaptiveMaxPool2d((1,1))
    self.drop_out2 = nn.Dropout(0.1)
    #self.linear = nn.Linear(32*block.expansion, 100)
    self.linear = nn.Linear(256, 100)
    
  def make_layer(self, block, out_channels, layer_len, stride = 1):
    downsample = None
    if (stride != 1) or (self.in_channels != out_channels):
      downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels, kernel_size = 1, stride = stride),
                                nn.BatchNorm2d(out_channels))
    
    layer = []
    layer.append(block(self.in_channels, out_channels, stride, downsample))
    self.in_channels = out_channels
    for i in range(1, layer_len):
      layer.append(block(out_channels, out_channels))
    return nn.Sequential(*layer)
  
  def forward(self, x):
    out = self.conv(x)
    #print('First convolution: \t', out.size())
    out = self.bn(out)
    out - self.relu(out)
    out = self.drop_out(out)
    
    out = self.layer1(out)
    out = self.drop_out(out)
    #print('First layer: \t\t', out.size())
    
    out = self.layer2(out)
    out = self.drop_out(out)
    #print('Second layer: \t\t', out.size())
    
    out = self.layer3(out)
    out = self.drop_out(out)
    #print('Second layer: \t\t', out.size())
    
    out = self.layer4(out)
    out = self.drop_out(out)
    #print('Second layer: \t\t', out.size())
    
    out = self.maxpool(out)
    out = self.drop_out2(out)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return(out)
 
#Calling the model:
layers = [2, 4, 4, 2]
model = ResNet(BasicBlock, layers)

for param in model.parameters():
    tensor0 = param.data
    dist.all_reduce(tensor0, op = dist.reduce_op.SUM)
    param.data = tensor0/np.sqrt(np.float(num_nodes))

#Enabling GPU:
model = model.cuda()

#Initializing loss function, optimizer and scheduler:
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum = 0.85, lr = learn_rate)

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
    for b_idx, (images, labels) in enumerate(train_loader):
        #print('inside training for loop')
        if (b_idx % num_nodes != rank):
            continue
        
        #print('loading image')
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        #print('finished loading image')
        
        optimizer.zero_grad()
        #print('finished optimizer zero grad')
        
        outputs = model(images)
        #print('created model')
        #optimizer.zero_grad()
        
        loss = loss_fn(outputs, labels)
        #print('created loss function')
        
        loss.backward()
        #print('finished loss backward')
        
        for param in model.parameters():
            tensor0 = param.grad.data.cpu()
            dist.all_reduce(tensor0, op = dist.reduce_op.SUM)
            tensor0 /= np.float(num_nodes)
            param.grad.data = tensor0.cuda()
        
        #print('finished sync for loop')
        optimizer.step()
        #print('finished optimizer step')
        
        _, predicted = torch.max(outputs.data, 1)
        #total = total + labels.size(0)
        correct = correct + (predicted == labels.data).sum()
        
        
    correct = comm.allreduce(correct, op = MPI.SUM)
    train_acc = correct/np.float(len(train_loader.dataset))
    if rank == 0:
        print('Training accuracy: \t', train_acc)
        train_acc_list.append(train_acc)
    
    #Testing:
    model.eval()
    correct = 0
    total = 0
    for b_idx, (images, labels) in enumerate(test_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total = total + labels.size(0)
        correct = correct + (predicted == labels.data).sum()
    
    correct = comm.allreduce(correct, op = MPI.SUM)
    test_acc = correct/np.float(len(test_loader.dataset))
    if rank == 0:
        print('Test Accuracy: \t\t', test_acc)
        print('**************************************************')
        test_acc_list.append(test_acc)
    model.train()


#print(train_acc_list)
#print(test_acc_list)



















