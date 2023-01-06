#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: Muhammed Balogun
# DATE CREATED: 23-12-2022                                 
# REVISED DATE: 

##
# Imports python modules
import argparse

def train_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. 
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create command line arguments
    parser.add_argument('--data_dir', type=str, default='flowers/', 
                        help='path to folder of images')
    parser.add_argument('--save_dir', type=str, default='save_directory', 
                        help='path to folder for saving training models')
    parser.add_argument('--arch', type=str, default = 'vgg19',
                        help='the CNN model architecture')
    parser.add_argument('--learning_rate', type=float, default = 0.001,
                        help='the learning rate of the training model')
    parser.add_argument('--epochs', type=int, default = 5,
                        help='the number of epochs for the training model')
    parser.add_argument('--hidden_units1', type=int, default = 4096,
                        help='the hidden units for the training model')
    parser.add_argument('--hidden_units2', type=int, default = 256,
                        help='the hidden units for the training model')
    parser.add_argument('--gpu', type=str, default = "gpu",
                        help='Use gpu for training model')
    
    return parser.parse_args()


def predict_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. 
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create command line arguments
    parser.add_argument('--data_dir', type=str, default='flowers/', 
                        help='path to folder of images')
    parser.add_argument('--save_dir', type=str, default='save_directory', 
                        help='path to folder for saving training models')
    parser.add_argument('--arch', type=str, default = 'vgg19',
                        help='the CNN model architecture')
    parser.add_argument('--category_names', type=str, default = 'cat_to_name.json', 
                        help='json file of names of flowers')
    parser.add_argument('--top_k', type=int, default = 5,
                        help='specify top class probability for prediction')
    parser.add_argument('--gpu', type=str, default = "gpu",
                        help='Use gpu for model prediction')
    parser.add_argument('--image_path', type=str, default = "flowers/test/71/image_04482.jpg",
                        help='Path to image used for prediction')
    
    return parser.parse_args()