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


def main():
    # 检测是否有可用的GPU，有则使用，否则使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device, " will be used.")
    data_dir_train = r"./datafiles_creating/all_raw_filelist.txt"
    data_dir_val = r"./datafiles_creating/val_raw_filelist.txt"
    data_dir_test = r"./datafiles_creating/test_raw_filelist.txt"

    resume = "best_checkpoint.pth.tar"
    # resume = None
    evaluate = True
    dataset = GcDataset(data_dir_train, data_dir_val, data_dir_test)
    print("Dataset created.")
    train_dataset = dataset.train_dataset
    val_dataset = dataset.val_dataset
    test_dataset = dataset.test_dataset

    batch_size = 800

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print("DataLoader created.")
    # 定义模型
    # model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    # 修改最后的全连接层以匹配CIFAR10的类别数（10类）
    # model.fc = nn.Linear(model.fc.in_features, 2)
    model = gcNet(dropout_rate=0.5)
    # # 如果有多个GPU，使用DataParallel来并行化模型
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    model.to(device)
    print("Model created.")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,weight_decay=1e-4)

    start_epoch = 0
    if resume is not None:
        print("Loading checkpoint info.")
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        best_acc = checkpoint['best_acc']
        print("Checkpoint loaded.")
    else :
        best_acc = 0



    # 记录标量值
    if evaluate is not True:
        writer = SummaryWriter('/logdir')
        # 训练模型
        num_epochs = 100
        print("Start training.")
        global_step = 0
        for epoch in range(start_epoch,num_epochs):
            total = 0
            model.train()
            correct = 0
            loss = torch.tensor(0.0)
            for images, labels in tqdm(train_loader,desc= "Training",leave=True):
                global_step += 1
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                writer.add_scalar('loss/train', loss, global_step)
                writer.add_scalar('accuracy/train', 100 * correct / total, global_step)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {100 * correct / total} %')


            # 在验证集上评估模型
            model.eval()
            with torch.no_grad():
                correct = 0
                total_val = 0
                for images, labels in tqdm(val_loader,desc="Validating",leave=True):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct += (predicted == labels).sum().item()

                acc = 100 * correct / total_val

                print(f'Accuracy of the model on the 4669 val images: {acc} %')
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': loss,
                    'best_acc': best_acc
                }
                if acc > best_acc:
                    best_acc = acc
                    torch.save(state, './best_checkpoint.pth.tar')
                else:
                    torch.save(state, './checkpoint.pth.tar')
                if acc>90:
                    print("Accuracy is higher than 90%, training will stop.")
                    break
        writer.close()

    model.eval()
    with torch.no_grad():
        correct = 0
        total_test = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Accuracy of the model on the 11649 test images: {100 * correct / total_test} %')

    print("Done!")

if __name__ == '__main__':
    main()