import os
from glob import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from skimage.transform import resize, rescale


def save_nii(arr, path):
    nii_img = nib.Nifti1Image(arr, affine=np.eye(4))
    nib.save(nii_img, path)


def load_nii(path_file):
    proxy = nib.load(path_file)
    array = proxy.get_fdata()
    return array


def get_dataset(path, filename):
    dataset = {}
    for path_file in sorted(glob(path+'/*/'+filename)):
        patient_id = path_file.split('/')[-2]
        array = load_nii(path_file)
        dataset[patient_id] = array
    return dataset


class MRI_7t(Dataset):
    def __init__(self, path_dataset):
        self.path_dataset = path_dataset

    def __len__(self):
        return len(self.path_dataset)

    def __getitem__(self, index):
        path_data = self.path_dataset[index]
        patient_id = path_data.split('/')[-1]

        # path_3t = f'{path_data}/3t_02_norm.nii'
        # path_7t = f'{path_data}/7t_02_norm.nii'

        path_3t = f'{path_data}/3t_02_norm_crop.nii'
        path_7t = f'{path_data}/7t_02_norm_crop.nii'

        x = nib.load(path_3t).get_data()
        y = nib.load(path_7t).get_data()

        x_h, x_w, x_d = x.shape
        y_h, y_w, y_d = y.shape

        x = torch.from_numpy(x).float().view(1, x_h, x_w, x_d)
        y = torch.from_numpy(y).float().view(1, y_h, y_w, y_d)

        return {'patient_id': patient_id, 'x': x, 'y': y}

