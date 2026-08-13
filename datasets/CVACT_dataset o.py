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



class CVACT_dataset_cropped(Dataset):
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

        if (self.is_train):
            self.T_lang = pd.read_csv(f'{self.path}/lang/{lang}_train.csv')
        else:
            self.T_lang = pd.read_csv(f'{self.path}/lang/{lang}_val.csv')


    def __len__(self):
        return len(self.data_csv)
    def __getitem__(self, item):
        anchor_image_name = self.str_images[item]
        anchor_image_path = f"{self.path}/streetview/{anchor_image_name}"
        # anchor_image_path = f"{self.path}/streetview_p/{anchor_image_name}"


        anchor_text = self.T_lang['Text'].loc[item]
        ###### Anchor Image #######
        anchor_img = Image.open(anchor_image_path).convert('RGB')
        # if self.is_train:
        # anchor_label = self.labels[item]
        # positive_list = self.index[self.index!=item][self.str_images[self.index!=item]==anchor_image_name]
        # positive_item = random.choice(positive_list)
        positive_image_name = self.sat_images[item]
        positive_image_path = f"{self.path}/satview_polish/{positive_image_name}"
        positive_img = Image.open(positive_image_path).convert('RGB')
        #positive_img = self.images[positive_item].reshape(28, 28, 1)
        negative_list = self.index[self.index!=item][self.sat_images[self.index!=item]!=positive_image_name]
        negative_item = random.choice(negative_list)
        negative_image_name = self.sat_images[negative_item]
        negative_image_path = f"{self.path}/satview_polish/{negative_image_name}"
        negative_img = Image.open(negative_image_path).convert('RGB')
        #negative_img = self.images[negative_item].reshape(28, 28, 1)

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
    

# class CVACT_dataset_cropped(Dataset):
#     def __init__(self, df,path, train=True, transform=None):
#         self.data_csv = df
#         self.is_train = train
#         self.transform = transform
#         self.path = path
#         if self.is_train:
#             self.sat_images = df.iloc[:, 0].values
#             self.str_images = df.iloc[:, 1].values
#             self.index = df.index.values 
#     def __len__(self):
#         return len(self.data_csv)
#     def __getitem__(self, item):
#         anchor_image_name = self.str_images[item]
#         anchor_image_path = f"{self.path}/{anchor_image_name}"
#         ###### Anchor Image #######
#         anchor_img = Image.open(anchor_image_path).convert('RGB')
#         if self.is_train:
#             # anchor_label = self.labels[item]
#             # positive_list = self.index[self.index!=item][self.str_images[self.index!=item]==anchor_image_name]
#             # positive_item = random.choice(positive_list)
#             positive_image_name = self.sat_images[item]
#             positive_image_path = f"{self.path}/{positive_image_name}"
#             positive_img = Image.open(positive_image_path).convert('RGB')
#             #positive_img = self.images[positive_item].reshape(28, 28, 1)
#             negative_list = self.index[self.index!=item][self.sat_images[self.index!=item]!=positive_image_name]
#             negative_item = random.choice(negative_list)
#             negative_image_name = self.sat_images[negative_item]
#             negative_image_path = f"{self.path}/{negative_image_name}"
#             negative_img = Image.open(negative_image_path).convert('RGB')
#             #negative_img = self.images[negative_item].reshape(28, 28, 1)
#             if self.transform!=None:
#                 anchor_img = self.transform(anchor_img)
#                 positive_img = self.transform(positive_img)                   
#                 negative_img = self.transform(negative_img)
#         return anchor_img, positive_img, negative_img
    


# class CVACT_Dataset_Eval(Dataset):
    
#     def __init__(self,
#                  data_folder,
#                  split,
#                  img_type,
#                  transforms=None,
#                  lang='T1'
#                  ):
        
#         super().__init__()
 
#         self.data_folder = data_folder
#         self.split = split
#         self.img_type = img_type
#         self.transforms = transforms
#         self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        
#         if split == 'train':
#             self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None)
#             if lang=='T1':
#                 self.df_lang = pd.read_csv(f'{data_folder}/lang/T1_train-19zl.csv')
#         else:
#             self.df = pd.read_csv(f'{data_folder}/splits/val-19zl.csv', header=None)
#             if lang=='T1':
#                 self.df_lang = pd.read_csv(f'{data_folder}/lang/T1_val-19zl.csv')

        
#         self.df = self.df.rename(columns={0:"sat", 1:"ground", 2:"ground_anno"})
        
#         self.df["idx"] = self.df.sat.map(lambda x : int(x.split("/")[-1].split(".")[0]))

#         self.idx2sat = dict(zip(self.df.idx, self.df.sat))
#         self.idx2ground = dict(zip(self.df.idx, self.df.ground))
   
    
#         if self.img_type == "reference":
#             self.images = self.df.sat.values
#             self.label = self.df.idx.values
            
#         elif self.img_type == "query":
#             self.images = self.df.ground.values
#             self.label = self.df.idx.values 
#         else:
#             raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")
                

#     def __getitem__(self, index):
        
#         # img = cv2.imread(f'{self.data_folder}/{self.images[index]}')
#         # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = Image.open(f'{self.data_folder}/{self.images[index]}').convert('RGB')
#         text = self.df_lang['Text'].loc[index]
        
#         # image transforms
#         if self.transforms is not None:
#             # img = self.transforms(img)
            
#             img = self.processor(images=img, return_tensors="pt")
#             img = img.pixel_values
#             img = torch.squeeze(img)

            
#         label = torch.tensor(self.label[index], dtype=torch.long)

#         return img, label, text


#         # if self.img_type == "query":    
#         #     return img, label, text
#         # else:
#         #     return img, label

#     def __len__(self):
#         return len(self.images)

            