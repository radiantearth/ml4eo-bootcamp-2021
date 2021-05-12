from torch.utils.data import Dataset
from torchvision import transforms, utils
import numpy as np
from scipy import ndimage
import torch

class ICLRDataset(Dataset):
    def __init__(self, imgs, areas, gts, field_masks, split_type, index):
        if split_type == 'test':
            idx = np.where(gts == -1)
            self.imgs = imgs[idx]
            self.areas = areas[idx]
            self.gts = gts[idx]
            self.field_masks = field_masks[idx]
        else:
            idx = np.array(index)
            self.imgs = imgs[gts > -1][idx]
            self.areas = areas[gts > -1][idx]
            self.field_masks = field_masks[gts > -1][idx]
            self.gts = gts[gts > -1][idx]            
        
        self.split_type = split_type
        self.feat_arr = [i for i in range(imgs.shape[2]) if i != 10] #remove B11 from features
        
    def __len__(self):
        return self.imgs.shape[0]
    
    def augment(self, img, mask):        
        p = np.random.random(3)
        ang = np.random.uniform(-10, 10)

        #remove data of randomly selected date (history augmentation)
        size = 1
        while True:
            idx_to_rmv = np.random.randint(low = img.shape[0], size = size).tolist()
            if np.unique(idx_to_rmv).shape[0] == size:
                break
        hist_idx = [i for i in range(img.shape[0]) if not (i in idx_to_rmv)]
        img = img[hist_idx]

        #apply flipping and rotation augmentation
        if p[1] > 0.5:
            mask[0] = np.flipud(mask[0])
        if p[2] > 0.5:
            mask[0] = ndimage.rotate(mask[0], ang, reshape = False)        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if p[1] > 0.5:
                    img[i,j] = np.flipud(img[i,j])
                if p[2] > 0.5:
                    img[i,j] = ndimage.rotate(img[i,j], ang, reshape = False)
        return img, mask

    def crop(self, img, mask):
        #randomly take a (16,16) crop from training image 
        size = 16
        while True:
            i = np.random.randint(0, 32 - size)
            j = np.random.randint(0, 32 - size)
            if mask[0, i:i+size, j:j+size].sum() > 0:
                break
        return img[:,:, i:i+size, j:j+size], mask[0, i:i+size, j:j+size]

    def __getitem__(self, idx):
        img = self.imgs[idx]
        field_mask = self.field_masks[idx]
        if self.split_type == 'train':
            img, field_mask = self.augment(img, field_mask)
            img, field_mask = self.crop(img, field_mask)
        return torch.FloatTensor(img[:, self.feat_arr]), torch.FloatTensor(self.areas[idx:idx+1]), torch.FloatTensor(field_mask), self.gts[idx]
