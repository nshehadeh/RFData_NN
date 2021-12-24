import numpy
import os
from torch.utils.data import Dataset
import torch
import warnings
import h5py
import math


class RFDataset(Dataset):
    def __init__(self, filename, num_samples, ignore=None, shuffle=True):  # target_is_data = False):
        self.filename = filename
        self.num_samples = num_samples
        self.shuffle = shuffle
        if not ignore:
            self.ignore = 0
        else:
            self.ignore = ignore
        # subtract 1 because of .dsstore file
        available_samples = len([name for name in os.listdir(str(filename) + "/input")]) # - 1

        # set num_samples
        if not num_samples:
            num_samples = available_samples

        # check to make sure given size is valid
        if num_samples > available_samples:
            warnings.warn('given data size is larger than available, overriding to make equal to available')
            self.num_samples = available_samples
        else:
            self.num_samples = num_samples
        #eval data
        size_num_samples = num_samples-ignore
        # FIXME change to take in as parameter
        fir_d = 41
        sec_d = 283
        # FIXME change to subtract number of ignored
        inputs = numpy.empty([size_num_samples, fir_d, sec_d])
        targets = numpy.empty([size_num_samples, fir_d, sec_d])
        
        track = size_num_samples-1
        for i in range(num_samples):
            if (i+1) != ignore:
                filename_input = filename + "/input/phantom_" + str(i+1) + "/chandat.mat"
                filename_target = filename + "/output_targets/phantom_" + str(i+1) + "/chandat.mat"
                if not os.path.isfile(filename_input):
                    raise IOError(filename_input + ' --> could not find input data file.')
                if not os.path.isfile(filename_target):
                    raise IOError(filename_target + ' --> could not find target data file.')
        
                fin = h5py.File(filename_input, 'r')
                ftar = h5py.File(filename_target, 'r')
                curr_input = fin['env']
                curr_targets = ftar['env']
                if curr_targets.shape[0] > fir_d:
                    curr_targets = curr_targets[:fir_d]
                if track >=  0:
                    print("New input @ index: " + str(track) + " from phantom: " + str(i+1))
                    inputs[track] = curr_input
                    targets[track] = curr_targets
                else:
                    print("Something went wrong in loading")
                
                track-=1

        self.data_tensor = torch.from_numpy(inputs)
        self.target_tensor = torch.from_numpy(targets)

    def __len__(self):
        return self.data_tensor.shape[0]*self.data_tensor.shape[1]

    def __getitem__(self, item):
        # images
        # return self.data_tensor[item], self.target_tensor[item]
        # slice in first dimension
        # return self.data_tensor[:][item], self.target_tensor[:][item]
        # slice in second dimension
        # return self.data_tensor[:][:][item], self.target_tensor[:][:][item]
        # one value?

        # compute channel
        # phantom # = floor(item / (1_d))
        # channel index = item % channel
        #
        p = math.floor(item / self.data_tensor.shape[1])
        # FIXME check this line
        c = item % self.data_tensor.shape[1]
        return self.data_tensor[p, c], self.target_tensor[p, c]

    def __str__(self):
        return f"RFDataset( {self.fname} )" + "\n" + f"{self.num_samples} / {self.available_samples}" + "\n"
