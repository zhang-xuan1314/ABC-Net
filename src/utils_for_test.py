import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from skimage import morphology


class MolecularImageDataset(Dataset):
    def __init__(self, df, transform=None):
        super(MolecularImageDataset, self).__init__()
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx,'name']
        path = '../data/UOB/processed_images/'+ path + '.png'

        temp_img = cv2.imread(path,flags=0)
        temp_img = (temp_img/255).astype('float32')
        temp_img = (temp_img>0.2)
        temp_img = 1-temp_img

        img = np.zeros([1,512,512]).astype('float32')
        img[0] = temp_img
        return img

def collate_fn(batch):
    """
    Collate to apply the padding to the captions with dataloader
    """
    imgs = batch
    imgs = [np.expand_dims(img,0) for img in imgs]
    imgs = np.concatenate(imgs,axis=0)
    imgs = torch.from_numpy(imgs)

    return imgs





