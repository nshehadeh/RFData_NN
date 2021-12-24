import os
import warnings
import time
import argparse

import torch
import yaml
from pprint import pprint

from dataloader import RFDataset
from model import FullyConnectedNet
from logger import Logger
from trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_params_path', help = 'Load params from a file')
    args = parser.parse_args()
    # print(args)
    # model_params_path = input("Enter config file path: ")
    # read model params from yaml config file (hard coded but replace with input (maybe args parser)
    config_stream = open(args.model_params_path, 'r')
    # config_stream = open('config_test', 'r')
    model_params = yaml.safe_load(config_stream)
    
    # cuda
    print('torch.cuda.is_available(): ' + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print('Using ' + str(torch.cuda.get_device_name(0)))
    else:
        print('Not using CUDA')
        model_params['cuda'] = False
    device = torch.device("cuda:0" if model_params['cuda'] else "cpu")

    num_samples = model_params['num_samples_train']
    dat_list = []
    print("Loading delayed channel data...")
    
    # for now, eval will be last sample
    num_eval = 1

    data_dir_path = model_params['data_date']
    num_datasets = len(data_dir_path)
    # dataloader is date folder, dataset
    for i in range(num_datasets):
        curr_dat = RFDataset('/data/beam_lab/shehadng/training_data/' + str(data_dir_path[i]), num_samples, num_eval)
        dat_list.append(curr_dat)
    print("Training data loaded: " + str(len(dat_list)))
    dat_train = torch.utils.data.ConcatDataset(dat_list)

    # FIXME strategies for rotating which you leave out, leave k out strategy
    # training data, currently ignoring first phantom for eval

    # FIXME cross validation strategy --> look into this
    # eval data

    dat_list = []
    for i in range(num_datasets):
        # FIXME change back curr_dat = RFDataset(num_phantoms-i, str(data_dir_path) + '/training_data', num_samples)
        curr_dat = RFDataset('/data/beam_lab/shehadng/training_data/' + str(data_dir_path[i]), num_eval, 0)
        dat_list.append(curr_dat)
    dat_eval = torch.utils.data.ConcatDataset(dat_list)
    print("Eval data loaded: " + str(len(dat_list)))

    dat_list = []
    # validation data
    for i in range(num_datasets):
        curr_dat = RFDataset('/data/beam_lab/shehadng/testing_data/' + str(data_dir_path[i]),  num_samples, 0)
        dat_list.append(curr_dat)
    dat_val = torch.utils.data.ConcatDataset(dat_list)
    print("Test data loaded: " + str(len(dat_list)))
    
    # drop last or not
    drop_last = False
    if (len(dat_train) % model_params['batch_size']) == 1:
        drop_last = True
        print("Drop last incomplete batch \n")

    # set up data loaders
    loader_train = torch.utils.data.DataLoader(dat_train, batch_size=model_params['batch_size'], shuffle=True,
                                               num_workers=1, drop_last=drop_last)

    loader_train_eval = torch.utils.data.DataLoader(dat_eval, batch_size=len(dat_eval), shuffle=False,
                                                    num_workers=1, drop_last=drop_last)

    loader_val = torch.utils.data.DataLoader(dat_val, batch_size=len(dat_val), shuffle=False,
                                             num_workers=1, drop_last=drop_last)
    # create model
    model = FullyConnectedNet(input_dim=model_params['input_dim'], output_dim=model_params['output_dim'],
                              layer_width=model_params['layer_width'], num_hidden=model_params['num_hidden'],
                              dropout=model_params['dropout'], dropout_input=model_params['dropout_input'])

    # loss fctn, for now just using L2 MSE
    # loss = "something"

    # optimizer, scheduler for now using Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=model_params['lr'], betas=(model_params['beta1'],
                                                                                   model_params['beta2']),
                                 weight_decay=model_params['weight_decay'])
    # start not using scheduler, use patience as fake scheduler = "fill in, start not using one
    # reduce learning rate on plateau
    # logger
    logger = Logger()

    # update model params
    model_params['num_samples_train'] = len(dat_train)
    model_params['num_samples_train_eval'] = len(dat_eval)
    model_params['num_samples_val'] = len(dat_val)
    model_params_path = os.path.join(model_params['save_dir'], 'model_params.txt')
    model_params['model_params_path'] = model_params_path

    # display input args
    print('\n')
    pprint(model_params)
    print('\n')

    # train
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      loader_train=loader_train,
                      patience=model_params['patience'],
                      loader_train_eval=loader_train_eval,
                      loader_val=loader_val,
                      cuda=model_params['cuda'],
                      logger=logger,
                      save_dir=model_params['save_dir'],
                      max_epochs=model_params['max_epochs'],
                      min_epochs=model_params['min_epochs'])
    trainer.train()
