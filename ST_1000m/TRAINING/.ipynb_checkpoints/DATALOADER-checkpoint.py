
from torchvision import transforms
from torch import nn
from torch import optim
import progressbar
#sys.path.append("/home2/datahome/tpicard/python/Python_Modules_p3_pyticles/")
import torch
from torch.utils.data import DataLoader, Dataset
from CNN_tools import *
from variables import *

# ## CREATE DATASET ####


class Pdf_Image_DataSet(Dataset):

    def __init__(self,images_norm, pdf,pdf_f,transform=None):

        self.pdf_f = pdf_f
        self.pdf = pdf
        self.images = images_norm
        self.transform = transform

    def __len__(self):
        return self.pdf_f.shape[0]

    def __getitem__(self, idx):
        # select coordinates
        #pos = idx%36
        #dt = idx//36

        pdf_f_sample = self.pdf_f[idx,:,:,:]
        pdf_sample = self.pdf[idx,:,:,:]
        images_norm_sample = self.images[idx,:,:,:]


        if self.transform:
            pdf_f_sample = self.transform(pdf_f_sample)
            pdf_sample = self.transform(pdf_sample)
            images_norm_sample = self.transform(images_norm_sample)

        return images_norm_sample, pdf_f_sample, pdf_sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        return torch.FloatTensor(sample)



class Pdf_Image_DataSet_OLD(Dataset):

    def __init__(self,images_norm, pdf,transform=None):

        self.pdf = pdf
        self.images = images_norm
        self.transform = transform

    def __len__(self):
        return self.pdf.shape[0]

    def __getitem__(self, idx):
        # select coordinates
        #pos = idx%36
        #dt = idx//36
        pdf_sample = self.pdf[idx,:,:,:]
        images_norm_sample = self.images[idx,:,:,:]


        if self.transform:
            pdf_sample = self.transform(pdf_sample)
            images_norm_sample = self.transform(images_norm_sample)

        return images_norm_sample, pdf_sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        return torch.FloatTensor(sample)
