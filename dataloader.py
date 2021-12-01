import numpy as np
import os
from torch.utils.data import Dataset
import torch
import warnings
import scipy.io


class RFDataset(Dataset):
    def __init__(self, phantom_num, filename, num_samples, shuffle=True):  # target_is_data = False):
        self.filename = filename
        self.num_samples = num_samples
        self.shuffle = shuffle

        # self.target_is_data = target_is_data
        filename_input = filename + "/input/phantom_" + str(phantom_num) + "/chandat.mat"
        filename_target = filename + "/output_targets/phantom_" + str(phantom_num) + "/chandat.mat"
        if not os.path.isfile(filename_input):
            raise IOError(filename_input + ' --> could not find input data file.')
        if not os.path.isfile(filename_target):
            raise IOError(filename_target + ' --> could not find target data file.')

        fin = scipy.io.loadmat(filename_input)
        ftar = scipy.io.loadmat(filename_target)

        # get number of samples available
        # use RF data variable within chandat.mat
        available_samples = fin['rf_data'].shape[0]

        # set num_samples
        if not num_samples:
            num_samples = available_samples

        # check to make sure given size is valid
        if num_samples > available_samples:
            warnings.warn('given data size is larger than available, overriding to make equal to available')
            self.num_samples = available_samples
        else:
            self.num_samples = num_samples

        # set inputs and targets
        inputs = np.hstack(fin['env'])
        # print("Inputs found and set to size: " + str(inputs.size))
        targets = np.hstack(ftar['env'])
        # print("Targets found and set to size: " + str(targets.size))

        self.data_tensor = torch.from_numpy(inputs).float()
        self.target_tensor = torch.from_numpy(targets).float()

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, item):
        return self.data_tensor[item], self.target_tensor[item]

    def __str__(self):
        return f"RFDataset( {self.fname} )" + "\n" + f"{self.num_samples} / {self.available_samples}" + "\n"
