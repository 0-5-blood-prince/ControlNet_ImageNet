import PIL
import PIL.Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
import numpy as np
import random
import cv2
from classes import IMAGENET2012_CLASSES
import json
import numpy as np
import cv2
from torch.utils.data import Dataset
import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class CustomDataset(Dataset):
    def __init__(self, photo_dir, sketch_dir, threshold=127, max_value=255, structuring_element_shape=cv2.MORPH_RECT, kernel_size=(5, 5)):
        self.photo_dir = photo_dir
        self.sketch_dir = sketch_dir
        self.category_list = np.array(sorted(os.listdir(photo_dir)))
        self.photo_file_list = []
        for category in self.category_list:
          photos = np.array(os.listdir(photo_dir + "/" + category))
          self.photo_file_list.extend(photos)
        self.sketch_file_list = []
        for category in self.category_list:
          sketches = np.array(os.listdir(sketch_dir + "/" + category))
          self.sketch_file_list.extend(sketches)
        self.threshold = threshold
        self.max_value = max_value
        self.kernel = cv2.getStructuringElement(structuring_element_shape, kernel_size)
        print(len(self.photo_file_list))
        print(len(self.sketch_file_list))
    
    def __len__(self):
        return len(self.photo_file_list)
    
    def __getitem__(self, idx):
        category = IMAGENET2012_CLASSES[self.photo_file_list[idx].split('_')[0]].split(',')[0].replace(' ','_')
        photo_name = os.path.join(self.photo_dir, category, self.photo_file_list[idx])
        sketch_name = os.path.join(self.sketch_dir, category, self.sketch_file_list[idx])
        source = cv2.imread(sketch_name)
        target = cv2.imread(photo_name)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = cv2.resize(target, (256,256), interpolation=cv2.INTER_AREA)
        _, binary_image = cv2.threshold(source, self.threshold, self.max_value, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary_image, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        sketch = cv2.bitwise_and(eroded, eroded, mask=cv2.cvtColor(target, cv2.COLOR_RGB2GRAY))
        source = sketch.astype(np.float32)
        target = target.astype(np.float32)
        categories = IMAGENET2012_CLASSES[self.photo_file_list[idx].split('_')[0]].split(',')
        text = "A real world image of " + category
        return target, photo_name, source, sketch_name, category


photo_path = '/home/ag4797/data/imagenet-1k/data/train/photo/'
photo_path_1 = '/home/ag4797/data/imagenet-1k/data/train_1/photo/'
sketch_path = '/home/ag4797/data/imagenet-1k/data/train/sketch/'
sketch_path_1 = '/home/ag4797/data/imagenet-1k/data/train_1/sketch/'


batch_size = 8
dataset = CustomDataset(photo_path, sketch_path)
photo, photo_name, sketch, sketch_name, category = dataset[798080]
print(category)
if not os.path.isdir(photo_path_1+'/'+category):
 os.makedirs(photo_path_1+'/'+category[i])
 os.makedirs(sketch_path_1+'/'+category[i])
cv2.imwrite(photo_name.replace('train','train_1'), photo.numpy())
cv2.imwrite(sketch_name.replace('train','train_1'), sketch.numpy())
'''
dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size)

for photos, photo_names, sketches, sketch_names, category in dataloader:
    for i in range(len(photo_names)):
        if not os.path.isdir(photo_path_1+'/'+category[i]):
            os.makedirs(photo_path_1+'/'+category[i])
            os.makedirs(sketch_path_1+'/'+category[i])
        cv2.imwrite(photo_names[i].replace('train','train_1'), photos[i].numpy())
        cv2.imwrite(sketch_names[i].replace('train','train_1'), sketches[i].numpy())

'''
