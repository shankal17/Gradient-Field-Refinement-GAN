"""Dataset to contain high and low resolution gradient fields

Classes
-------
GradientDataset()
    Dataset containing high and low resolution gradients of a given surface function
"""

import os
import torch
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset

class GradientDataset(Dataset):
    """Discriminator, or classifier, based off of the SRGAN paper
    found here: https://arxiv.org/abs/1609.04802
    ...
    
    Attributes
    ----------
    data_folder : String
        Path of folder containing surface functions
    total_gradients : List
        List of sorted surface funtion path names
    split_type : String
        Descriptive of type of dataset
    downsample_factor : int
        Downsample factor of calculated low resolution gradient
        (must match generator upsample factor if training)
    
    Methods
    -------
    __getitem__(idx)
        Returns high and low resolution gradient fields of suface function at index idx
    """

    def __init__(self, data_folder, split_type, downsample_factor=4):
        self.data_folder = data_folder
        all_gradients = os.listdir(data_folder)
        self.total_gradients = natsorted(all_gradients)
        self.split_type = split_type.lower()
        self.downsample_factor = downsample_factor
        assert self.split_type in {'train', 'test', 'val'}

    def __getitem__(self, idx):
        """Returns high and low resolution gradient fields of suface function at index idx
        Parameters
        ----------
        idx : int
            Index of surface fuction path
        
        Returns
        -------
        torch.Tensor
            Surface function high resolution gradient field
        torch.Tensor
            Surface function low resolution gradient field
        """
        
        img_loc = os.path.join(self.data_folder, self.total_gradients[idx])
        hr_grad = np.load(img_loc)[-2:]
        lr_grad = np.array([comp[::self.downsample_factor, ::self.downsample_factor] for comp in hr_grad])
        # print(hr_grad.shape)
        # print(lr_grad.shape)
        hr_grad = torch.from_numpy(hr_grad)
        lr_grad = torch.from_numpy(lr_grad)
        return hr_grad, lr_grad

    def __len__(self):
        return len(self.total_gradients)
