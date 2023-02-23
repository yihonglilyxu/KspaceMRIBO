import torch
import torch.fft
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import matplotlib.pyplot as plt
import numpy as np
import random 
from torchvision.datasets import ImageFolder
import re
from torch.utils.data import Dataset, DataLoader
import h5py

import fastmri 
from fastmri.fftc import fft2c_new, ifft2c_new

#device = "cuda" if torch.cuda.is_available() else "cpu"
#kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}

#batchsize=16
#k-space image size batchx256x256x2 already resized and normalized 

class MRIDataset_h5(torch.utils.data.Dataset):
    def __init__(self, in_file, mode = None):
        super(MRIDataset_h5, self).__init__()
#        self.in_file = in_file
        self.mode = mode
        self.file = h5py.File(in_file, 'r')
        self.n_images = len(self.file.keys())

    def __len__(self):
        
        return self.n_images
    
    def __getitem__(self, index):
        input = self.file[str(index)][()]
        
        if self.mode == 'real_imag':
            pass
        
        elif self.mode == 'mag_phase': 
            input = torch.view_as_complex(torch.from_numpy(input))
            input = torch.cat((input.abs(),input.angle()),dim = 2)
         
       
        return input

