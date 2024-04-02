from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import os
import torch

class CustomGcDataset(Dataset):
    def __init__(self,text_file, transform=None):
        """
        初始化数据集。
        :param txt_file: 包含图像路径和标签的txt文件。
        :param transform: 应用于每个图像的转换。
        """
        self.img_labels = []
        self.transform = transform


        with open(text_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) == 2:
                    img_path, label = parts
                    self.img_labels.append((img_path,int(label)))
                else:
                    raise ValueError("每行应该包含图像路径和标签!")


    def __len__(self):
        """
        返回数据集中图像的数量。
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        根据给定的索引idx，从数据集中获取一个样本。
        :param idx: 样本的索引。
        """
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path).convert('RGB')  # 转换图像到RGB，以防它不是RGB格式

        if self.transform:
            image = self.transform(image)


        return image,  torch.tensor(label)


class GcDataset(object):
    def __init__(self, data_dir_train,data_dir_val,data_dir_test):
        normalize = transforms.Normalize(mean=[0.6531, 0.4834, 0.4051], std=[0.2352, 0.2065, 0.1971])

        train_trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((90,120)),
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
             # 如果你使用的是PyTorch，这个过程可以通过transforms.ToTensor()实现，这个转换不仅将PIL图像或NumPy数组转换为FloatTensor，并且缩放像素值到0到1的范围。
            # normalize,
        ])

        test_trsfm = transforms.Compose([
            transforms.Resize((90,120)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        if data_dir_train and os.path.isfile(data_dir_train):
            self.train_dataset = CustomGcDataset(data_dir_train, transform=train_trsfm)
        else:
            self.train_dataset =None
            print("=>warning! Not found train")

        if data_dir_val and os.path.isfile(data_dir_val):
            self.val_dataset = CustomGcDataset(data_dir_val, transform=test_trsfm)
        else:
            self.val_dataset =None
            print("=>warning! Not found val")

        if data_dir_test and os.path.isfile(data_dir_test):
            self.test_dataset = CustomGcDataset(data_dir_test, transform=test_trsfm)
        else:
            self.test_dataset =None
            print("=>warning! Not found test")



