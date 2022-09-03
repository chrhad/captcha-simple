#!/usr/bin/env python
# Author: Christian Hadiwinoto
# Dataset and transform functions
import math
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Union, Dict
import os

from utils import LABEL_IDXS

class ExtractImagePixelSimple():
    """ Extract features with simple algorithm and output tensor
    Assumes to same input size 60x30 px; label 5 sequence

    Args:
        sample (dict): {"image": image path .jpg, "label": label path .txt}
    
    Returns:
        tensorized_sample (Dict[str,torch.Tensor])
    """
    def __call__(self, sample: Union[str, Dict[str, str]]) -> Union[torch.Tensor,Dict[str, torch.Tensor]]:
        if not isinstance(sample, Dict):
            return self.extract_image_feat(sample)  # input str -> output torch.Tensor
        output = {'image': self.extract_image_feat(sample['image'])}
        if 'label' in sample:
            output['label'] = self.extract_label(sample['label'])
        return output # input dict -> output dict

    def extract_image_feat(self, image_path):
        """ extract image pixels and convert to tensor

        Args:
            image_path (str): path to image file in .jpg format
        
        Returns:
            tensorized_image (torch.Tensor)
        """
        im = Image.open(image_path)
        height, width = im.size[1], im.size[0]
        pix = im.load()  # Get the RGBA values of the pixels of an image

        # tensor sized width x height because the sequence is along the width and the features is along the height
        # this looks like word embeddings with dimension sizes: length x emb_size
        return torch.tensor([
            [1-(sum(pix[x,y]) / 765) for y in range(height)] for x in range(width)])
    
    def extract_label(self, label_path):
        """ extract label from the specified path
        
        Args:
            label_path (str): path to label file in .txt format
        
        Returns:
            tensorized_label_ids (torch.Tensor)
        """
        with open(label_path, 'r', encoding='utf-8') as f:
            line = f.read().strip()
            return torch.tensor([LABEL_IDXS[c] for c in line], dtype=torch.long)


class ImageDataset(Dataset):
    def __init__(self, path_list, transform=None):
        """ Initialize image dataset from a list of dict with keys "image" and "label"

        Args:
            path_list (List[str])
        """
        self.path_list = path_list
        self.transform = transform
    
    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.path_list[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    

def get_label(input_fname, label_dir):
    # input_fname MUST start with 'input'
    output_path = os.path.join(label_dir, 'output' + input_fname[5:].replace('.jpg', '') + '.txt')
    if os.path.exists(output_path):
        return output_path
    return None


def split_traindev(input_dir, label_dir, heldout=0.2):
    """ Automatically detect image and label files to construct training and development set
    Skips those without corresponding label

    Args:
        input_dir (str): directory name where image files are stored inputNN.jpg
        label_dir (str): directory name where label files are stored outputNN.jpg
        heldout (float): portion of data to be heldout for devset, default = 0.1
    
    Returns:
        train_dataset (ImageDataset)
        dev_dataset (ImageDataset)
    """
    all_df = pd.DataFrame(sorted([(os.path.join(input_dir, fname), get_label(fname, label_dir)) for fname in os.listdir(input_dir) \
        if fname.startswith('input') and fname.endswith('.jpg') and get_label(fname, label_dir)]), columns=["image", "label"])
    all_len = len(all_df)
    dev_end = math.ceil(heldout * all_len)
    train_list, dev_list = all_df[dev_end:].to_dict('records'), all_df[:dev_end].to_dict('records')
    return ImageDataset(train_list, ExtractImagePixelSimple()), ImageDataset(dev_list, ExtractImagePixelSimple())


if __name__ == '__main__':
    train_dset, dev_dset = split_traindev('data/input', 'data/output')
    print("Training dataset size: {:d}".format(len(train_dset)))
    for (i, sample) in enumerate(train_dset):
        print("{:d}: {}".format(i, sample))
    print("Development dataset size: {:d}".format(len(dev_dset)))
    for (i, sample) in enumerate(dev_dset):
        print("{:d}: {}".format(i, sample))
