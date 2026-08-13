import os
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import json
from torchvision.transforms import transforms
import random
from transformers import CLIPProcessor, AutoProcessor, AutoTokenizer
from attributes import Configuration as hypm



class VIGOR_dataset_cropped(Dataset):
    def __init__(self, df, path, train=True, transform=None, lang='T1'):
        self.data_csv = df
        self.is_train = train
        self.transform = transform
        self.path = path
        self.lang = lang
        self.tokenizer = AutoTokenizer.from_pretrained(hypm.t_pretrain_weight, use_fast=True)
        self.processor = AutoProcessor.from_pretrained(hypm.v_pretrain_weight, use_fast=True)

        # if self.is_train:
        self.sat_images = df.iloc[:, 1].values
        self.str_images = df.iloc[:, 0].values
        self.index = df.index.values
        self.data_csv["idx"] = self.data_csv.index



    def __len__(self):
        return len(self.data_csv)
    def __getitem__(self, item):
        anchor_image_name = self.str_images[item]
        anchor_image_path = f"{self.path}/{anchor_image_name}"

        anchor_text = self.data_csv['T1_response'].loc[item]
        ###### Anchor Image #######
        anchor_img = Image.open(anchor_image_path).convert('RGB')
        positive_image_name = self.sat_images[item]
        positive_image_path = f"{self.path}/{positive_image_name}"
        positive_img = Image.open(positive_image_path).convert('RGB')
        
        negative_list = self.index[self.index!=item][self.sat_images[self.index!=item]!=positive_image_name]
        negative_item = random.choice(negative_list)
        negative_image_name = self.sat_images[negative_item]
        negative_image_path = f"{self.path}/{negative_image_name}"
        negative_img = Image.open(negative_image_path).convert('RGB')


        if self.transform!=None:
            # anchor_img = self.transform(anchor_img)
            # positive_img = self.transform(positive_img)                   
            # negative_img = self.transform(negative_img)

            anchor_img = self.processor(images=anchor_img, return_tensors="pt")
            positive_img = self.processor(images=positive_img, return_tensors="pt")
            negative_img = self.processor(images=negative_img, return_tensors="pt")
            # anchor_text = self.tokenizer(anchor_text, padding=True, return_tensors="pt", max_length=77, truncation=True)
            

            anchor_img = anchor_img.pixel_values
            anchor_img = torch.squeeze(anchor_img)

            positive_img = positive_img.pixel_values
            positive_img = torch.squeeze(positive_img)

            negative_img = negative_img.pixel_values
            negative_img = torch.squeeze(negative_img)





        return anchor_img, positive_img, negative_img, anchor_text, self.data_csv.idx[item]
    

