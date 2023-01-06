#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PROGRAMMER: Muhammed Balogun 
# DATE CREATED: 01-12-2022                               
# REVISED DATE: 06-01-2023
# PURPOSE: Train flower images using a pretrained CNN model, compares these
#          classifications to the true identity of the flower in the images.


# import python modules
import numpy as np
import matplotlib.pyplot as plt
import argparse

import os
from workspace_utils import active_session

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

# Imports functions created for this program
from get_input_args import train_args
from transforms_data import transforms_data

# Define input arguments
args = train_args()
data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
hidden_units1 = args.hidden_units1
hidden_units2 = args.hidden_units2
epochs = args.epochs
lr = args.learning_rate
gpu = args.gpu


def train_model():
    
    # Read in train and validation data
    trainloader, validloader, testloader, train_data = transforms_data(data_dir)
    
    # Use GPU if it's available
    if torch.cuda.is_available() and gpu==gpu:
        device = torch.device("cuda")
      
    else:
        device = torch.device("cpu")
    
    # Define training model
    if arch == "vgg19":
        model = models.vgg19(pretrained=True)
        
    elif arch =="vgg16":
        model = models.vgg16(pretrained=True)
        
    elif arch =="vgg13":
        model = models.vgg13(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Customize model classifier
    classifier = nn.Sequential(
        OrderedDict([
            ('fc1', nn.Linear(25088, hidden_units1)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.2)),
            ('fc2', nn.Linear(hidden_units1, hidden_units2)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.2)),
            ('fc3', nn.Linear(hidden_units2, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ])
    )

    model.classifier = classifier
    
    # Define loss function
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model.to(device);

    with active_session():

        steps = 0
        running_loss = 0
        print_every = 10
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
                
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                logps = model.forward(inputs)
                loss = criterion(logps, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            valid_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                          f"Valid accuracy: {accuracy/len(validloader):.3f}")
                    running_loss = 0
                    model.train()

    # validation on the test set
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Test loss: {test_loss/len(testloader):.3f}.. "
              f"Test accuracy: {accuracy/len(testloader):.3f}")

    
    # Save the checkpoint
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'input_size': 25088,
        'output_size': 102,
        'epochs': epochs,
        'classifier': model.classifier,
        'optimizer': optimizer.state_dict(),
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))
    
if __name__ == "__main__":
    train_model()