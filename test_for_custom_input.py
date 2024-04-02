import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# from torchvision.models import resnet18
from gcNet import gcNet
from torch.utils.data import DataLoader
from gc_loader import GcDataset
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

normalize = transforms.Normalize(mean=[0.6531, 0.4834, 0.4051], std=[0.2352, 0.2065, 0.1971])


test_trsfm = transforms.Compose([
    transforms.Resize((90, 120)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
resume = "best_checkpoint.pth.tar"
model = gcNet(dropout_rate=0.5)
model.eval()
if resume is not None:
    print("Loading checkpoint info.")
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['model'])
    print("Checkpoint loaded.")

img_path ="./dataset/handman_test/male/img_2.png"
# img_path = r"N:\program-files\mia_programs\Gender_classification\dataset\archive\Validation\female\112973.jpg.jpg"
image = Image.open(img_path).convert('RGB')  # 转换图像到RGB，以防它不是RGB格式

if test_trsfm:
    image = test_trsfm(image)
print(image.shape)


image.reshape(1,3,90,120)
output =model(image.reshape(1,3,90,120) )
opt = output.argmax()
print(output)
if opt == 0:
    print("female")
else:
    print("male")
