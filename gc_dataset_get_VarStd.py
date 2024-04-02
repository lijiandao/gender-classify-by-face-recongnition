import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from gc_loader import GcDataset

from tqdm import tqdm
def calculate_mean_std(dataloader=None,input=None):
    # 用于累积所有图像的和，以及平方的和
    channel_sum, channel_squared_sum, num_batches = 0, 0, 0
    if input is None:
        for data, _ in tqdm(dataloader,desc="Calculating mean and std"):
            # data的形状为[batch_size, channels, height, width]
            channel_sum += torch.mean(data, dim=[0, 2, 3])
            channel_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
            num_batches += 1
    else:
        channel_sum += torch.mean(input, dim=[0, 2, 3])
        # 当我们在这三个维度上计算均值时，我们实际上是在对每个通道的所有图像的所有像素值进行累加，然后除以这些像素值的总数。这个总数是批次大小、高度和宽度的乘积。
        channel_squared_sum += torch.mean(input ** 2, dim=[0, 2, 3])
        # channel_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3]) data 是张量，怎么能**2？
        # 代表对每个元素进行平方，然后在这三个维度上计算均值
        num_batches += 1

    # 计算均值和标准差
    mean = channel_sum / num_batches
    # var = E[X^2] - (E[X])^2
    std = (channel_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

if __name__ == '__main__':
    data_dir_train = r"N:\program-files\mia_programs\Gender_classification\datafiles_creating\all_raw_filelist.txt"
    # data_dir_val = r"N:\program-files\mia_programs\Gender_classification\datafiles_creating\val_raw_filelist.txt"
    # data_dir_test = r"N:\program-files\mia_programs\Gender_classification\datafiles_creating\test_raw_filelist.txt"

    gcdataset = GcDataset(data_dir_train, None, None)
    train_dataset = gcdataset.train_dataset

    batch_size = 3
    # 假设你已经有了一个数据加载器
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    input = torch.randn(5, 3, 64, 64)
    mean, std = calculate_mean_std(trainloader,None)
    print("Mean: ", mean)
    print("Std: ", std)
#
# Mean:  tensor([0.6531, 0.4834, 0.4051])
# Std:  tensor([0.2352, 0.2065, 0.1971])
