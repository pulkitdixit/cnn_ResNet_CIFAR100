# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:46:08 2019

@author: Pulkit Dixit
"""

import numpy as np
import torch
import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
#from torch.autograd import Variable
import torch.distributed as dist


import os
import subprocess
from mpi4py import MPI

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

batch_size = 100
learn_rate = 0.001
scheduler_step_size = 8
scheduler_gamma = 0.5
num_epochs = 50

transform_train = transforms.Compose([transforms.RandomRotation(10),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()
                                     ])

train_dataset = torchvision.datasets.CIFAR100(root = '~/scractch', train=True, transform=transform_train, download=True)
test_dataset = torchvision.datasets.CIFAR100(root = '~/scractch', train=False, transform=transform_train, download=True)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = 8)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False, num_workers = 8)

def conv3x3(in_channels, out_channels, stride = 1):
  return nn.Conv2d(in_channels, out_channels, stride = stride, kernel_size = 3, padding = 1)

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
    
layers = [2, 4, 4, 2]
model = ResNet(BasicBlock, layers)

for param in model.parameters():
    tensor0 = param.data
    dist.all_reduce(tensor0, op = dist.reduce_op.SUM)
    tensor0 /= np.float(num_nodes)

use_cuda = True
if use_cuda and torch.cuda.is_available():
    model = model.cuda()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 
                                lr = learn_rate)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                            step_size = scheduler_step_size, 
                                            gamma = scheduler_gamma)

train_acc_list = []
test_acc_list = []
for epochs in range(num_epochs):
    scheduler.step()
    correct = 0
    total = 0
    print('Current epoch: \t\t', epochs+1, '/', num_epochs)
    #print('--------------------------------------------------')
    for images, labels in train_loader:
        #images = images.reshape(-1, 16*16)
        images = images
        labels = labels
        if use_cuda and torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        for params in model.parameters():
            tensor0 = param.grad.data.cpu()
            dist.all_reduce(tensor0, op = dist.reduce_op.SUM)
            tensor0 /= np.float(num_nodes)
            param.grad.data = tensor0.cuda()
        
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        #total = total + labels.size(0)
        correct = correct + (predicted == labels).sum().item()
        
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
        
    #train_acc = correct/total
    correct = comm.all_reduce(correct, op = MPI.SUM)
    train_acc = correct/np.float(len(train_loader.dataset))
    if rank == 0:
        print('Training accuracy: \t', train_acc)
    #print('--------------------------------------------------')
    train_acc_list.append(train_acc)
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images
            labels = labels
            if use_cuda and torch.cuda.is_available():
              images = images.cuda()
              labels = labels.cuda()
        
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()
    test_acc = correct/total
    print('Test Accuracy: \t\t', test_acc)
    print('**************************************************')
    test_acc_list.append(test_acc)
    model.train()




















