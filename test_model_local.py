#!/usr/bin/env python

import numpy as np
import os
import h5py
import scipy
import torch
import yaml
import numpy
import math
from torch import nn
import argparse
from scipy.io import savemat
import time

from dataloader import RFDataset
from model import FullyConnectedNet

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_params_path', help='Load params from a file')
    # parser.add_argument('--model_dir', help='Load model from directory')
    # args = parser.parse_args()
    # model_params_path = "config_model"
    print(os.listdir())
    model_dir = "saved_models/DNN_20hid/"
    date_dir = "20211216"
    config_stream = open('config_model', 'r')
    model_params = yaml.safe_load(config_stream)
    # model_dir = args.model_dir
    # cuda
    print('torch.cuda.is_available(): ' + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print('Using ' + str(torch.cuda.get_device_name(0)))
    else:
        print('Not using CUDA')
        model_params['cuda'] = False
    device = torch.device("cuda:0" if model_params['cuda'] else "cpu")

    # load test set
    print("Loading test data...")
    data_dir_path = model_params['data_date']
    num_datasets = len(data_dir_path)
    dat_list = []
    # validation data
    for i in range(num_datasets):
        curr_dat = RFDataset('testing_data/' + date_dir, 4, 0)
        dat_list.append(curr_dat)
    dat_val = torch.utils.data.ConcatDataset(dat_list)

    loader_val = torch.utils.data.DataLoader(dat_val, batch_size=len(dat_val), shuffle=False,
                                             num_workers=1)

    print("Data loaded for evaluation: " + str(len(dat_list)))
    model = FullyConnectedNet(input_dim=model_params['input_dim'],
                              output_dim=model_params['output_dim'],
                              layer_width=model_params['layer_width'],
                              num_hidden=model_params['num_hidden'],
                              dropout=model_params['dropout'],
                              dropout_input=model_params['dropout_input'], )
    print('Loading weights from: ' + model_dir + 'model.dat')
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.dat'), map_location='cpu'))
    model.eval()
    new_data = numpy.empty([4, 41, 283])
    temp_data = numpy.empty([205, 283])
    # tracker = 0
    for batch_size, data in enumerate(loader_val):
        print(data[0])
        inputs = data[0].float()
        print("Data dimensions: " + str(data[0].shape))
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            print(outputs)
            temp_data = outputs.numpy()
        # tracker += 1


    # write out new_data
    print("Going to print new data")
    for i in range(math.floor(temp_data.shape[0]/41)):
        print("From: " + str(41*i) + " to " + str(41*(i+1)))
        new_data[i][:][:] = temp_data[(41*i):(41*(i+1))][:]
        print(new_data[i])

    for i in range(4):
        scipy.io.savemat(model_dir + "Phantom_1/new_data" + str(i) + ".mat", {'new_env': new_data[i]})

