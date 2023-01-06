#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PROGRAMMER: Muhammed Balogun 
# DATE CREATED: 01-12-2022                               
# REVISED DATE: 06-01-2023
# PURPOSE: Predict flower images using a pretrained CNN model, compares these
#          classifications to the true identity of the flower in the images.

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

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
from get_input_args import predict_args

# Define input arguments
args = predict_args()
data_dir = args.data_dir
save_dir = args.save_dir
json_file = args.category_names
arch = args.arch
gpu = args.gpu
image_path = args.image_path
topk = args.top_k



# Function to load checkpoint and rebuild model
def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    
    # Define training model
    if arch == "vgg19":
        model = models.vgg19(pretrained=True)
        
    elif arch =="vgg16":
        model = models.vgg16(pretrained=True)
        
    elif arch =="vgg13":
        model = models.vgg13(pretrained=True)
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

model = load_checkpoint('save_directory/checkpoint.pth')


# Function for preprocessing image for prediction
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    image_transforms = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])
        ])
    
    # Apply transform to image
    image = image_transforms(pil_image)
    
    return image


def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Use GPU if it's available
    if torch.cuda.is_available() and gpu==gpu:
        device = torch.device("cuda")
      
    else:
        device = torch.device("cpu")
    
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    model.eval()

    # Convert 2D image to 1D vector
    img = process_image(image_path)
    img = img.unsqueeze(0)

    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        logps = model.forward(img.to(device))

    probs = torch.exp(logps)
    top_p, top_class = probs.topk(topk, dim=1)

    return top_p, top_class

def predict_image():
    
    with open(json_file, "r") as file:
        cat_to_name = json.load(file)
    
    image_process = process_image(image_path)
    probs, classes = predict(image_path, model, topk)
    image_labels = [cat_to_name[str(i)] for i in classes.cpu().numpy().tolist()[0]]
    
    print(np.array(probs[0]))
    print(image_labels)
          
if __name__ == "__main__":
    predict_image()