import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import torchvision
import cv2
from torchvision import transforms
import os
from random import randint
from pytorch_lightning import Trainer

# declaring the path of the train and test folders
train_path = "../Downloads/dataset/archive/DATASET/TRAIN"
test_path = "../Downloads/dataset/archive/DATASET/TEST"
classes_dir_data = os.listdir(train_path)
num_of_classes = len(classes_dir_data)
print("Total Number of Classes :" , num_of_classes)
num = 0
classes_dict = {}
num_dict = {}
for c in  classes_dir_data:
    # print(c)
    classes_dict[c] = num
    num_dict[num] = c
    num = num +1
"""
num_dict contains a dictionary of the classes numerically and it's corresponding classes.
classes_dict contains a dictionary of the classes and the coresponding values numerically.
"""
num_of_classes = len(classes_dir_data)
classes_dict

class Image_Dataset(Dataset):

    def __init__(self,classes,image_base_dir,transform = None, target_transform = None):

        """

        classes:The classes in the dataset

        image_base_dir:The directory of the folders containing the images

        transform:The trasformations for the Images

        Target_transform:The trasformations for the target

        """

        self.img_labels = classes

        self.imge_base_dir = image_base_dir

        self.transform = transform

        self.target_transform = target_transform

    def __len__(self):

        return len(self.img_labels)

    def __getitem__(self,idx):

        img_dir_list = os.listdir(os.path.join(self.imge_base_dir,self.img_labels[idx]))

        image_path = img_dir_list[randint(0,len(img_dir_list)-1)]

        #print(image_path)

        image_path = os.path.join(self.imge_base_dir,self.img_labels[idx],image_path)

        image = cv2.imread(image_path)

        if self.transform:

            image = self.transform(image)

        if self.transform:

            label = self.target_transform(self.img_labels[idx])

        return image,label
    
size = 50

basic_transformations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((size,size)),
        transforms.Grayscale(1),
    transforms.ToTensor()])
training_transformations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((size,size)),
    transforms.RandomRotation(degrees = 45),
    transforms.RandomHorizontalFlip(p = 0.005),
        transforms.Grayscale(1),
    transforms.ToTensor()
])

def target_transformations(x):
    return torch.tensor(classes_dict.get(x))


class YogaDataModule(pl.LightningDataModule):

    def __init__(self):

            super().__init__()            

    def prepare_data(self):

        self.train = Image_Dataset(classes_dir_data,train_path,training_transformations,target_transformations)

        self.valid = Image_Dataset(classes_dir_data,test_path,basic_transformations,target_transformations)

        self.test = Image_Dataset(classes_dir_data,test_path,basic_transformations,target_transformations)

    def train_dataloader(self):

        return DataLoader(self.train,batch_size = 64,shuffle = True)

    def val_dataloader(self):  

        return DataLoader(self.valid,batch_size = 64,shuffle = True)

    def test_dataloader(self):

        return DataLoader(self.test,batch_size = 64,shuffle = True)



class YogaModel(pl.LightningModule):

    def __init__(self):

        super().__init__()

        """

        The convolutions are arranged in such a way that the image maintain the x and y dimensions. only the channels change

        """

        self.layer_1 = nn.Conv2d(in_channels = 1,out_channels = 3,kernel_size = (3,3),padding = (1,1),stride = (1,1))

        self.layer_2 = nn.Conv2d(in_channels = 3,out_channels = 6,kernel_size = (3,3),padding = (1,1),stride = (1,1))

        self.layer_3 = nn.Conv2d(in_channels = 6,out_channels = 12,kernel_size = (3,3),padding = (1,1),stride = (1,1))

        self.pool = nn.MaxPool2d(kernel_size = (3,3),padding = (1,1),stride = (1,1))

        self.layer_5 = nn.Linear(12*50*50,1000)#the input dimensions are (Number of dimensions * height * width)

        self.layer_6 = nn.Linear(1000,100)

        self.layer_7 = nn.Linear(100,50)

        self.layer_8 = nn.Linear(50,10)

        self.layer_9 = nn.Linear(10,5)

    def forward(self,x):

        """

        x is the input data

        """

        x = self.layer_1(x)

        x = self.pool(x)

        x = self.layer_2(x)

        x = self.pool(x)

        x = self.layer_3(x)

        x = self.pool(x)

        x = x.view(x.size(0),-1)

        print(x.size())

        x = self.layer_5(x)

        x = self.layer_6(x)

        x = self.layer_7(x)

        x = self.layer_8(x)

        x = self.layer_9(x)

        return x

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(),lr = 1e-7)

        return optimizer

    """

    The Pytorch-Lightning module handles all the iterations of the epoch

    """

    def training_step(self,batch,batch_idx):

        x,y = batch

        y_pred = self(x)

        loss = F.cross_entropy(y_pred,y)

        return loss

    def validation_step(self,batch,batch_idx):

        x,y = batch

        y_pred = self(x)

        loss = F.cross_entropy(y_pred,y)

        return loss

    def test_step(self,batch,batch_idx):

        x,y = batch

        y_pred = self(x)

        loss = F.cross_entropy(y_pred,y)

        self.log("loss",loss)

        return loss
    

model = YogaModel()

module = YogaDataModule()

trainer = Trainer(max_epochs=5)

trainer.fit(model,module)

trainer.test(model, module.test_dataloader())

