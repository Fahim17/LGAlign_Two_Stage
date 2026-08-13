import random

from PIL import Image
from torch.utils.data import Dataset


class GAMa_dataset_cropped(Dataset):
    def __init__(self, df, path, train=True, transform=None, lang='T1'):
        self.data_csv = df
        self.is_train = train
        self.transform = transform
        self.path = path
        self.lang = lang

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
        anchor_img = Image.open(anchor_image_path).convert('RGB')

        positive_image_name = self.sat_images[item]
        positive_image_path = f"{self.path}/{positive_image_name}"
        positive_img = Image.open(positive_image_path).convert('RGB')

        negative_list = self.index[self.index != item][self.sat_images[self.index != item] != positive_image_name]
        negative_item = random.choice(negative_list)
        negative_image_name = self.sat_images[negative_item]
        negative_image_path = f"{self.path}/{negative_image_name}"
        negative_img = Image.open(negative_image_path).convert('RGB')

        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img, anchor_text, self.data_csv.idx[item]