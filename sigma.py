import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import resnet50, resnet34
import os
import json


class DLDataset(Dataset):
    def __init__(self, inds, test, labels, transform):
        super().__init__()
        self.transform = transform
        self.inds = inds
        self.is_test = test
        self.labels = labels

    def __getitem__(self, item):
        if self.is_test:
            image_path = "bhw1/test/test_" + f"{str(self.inds[item]).zfill(5)}" + ".jpg"
        else:
            image_path = "bhw1/trainval/trainval_" + f"{str(self.inds[item]).zfill(5)}" + ".jpg"
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        if self.is_test:
            return image, np.zeros(0)
        label = self.labels.iloc[[self.inds[item]], :]['Category']
        return image, np.array(label)

    def __len__(self):
        return self.inds.shape[0]


def my_train_val_test_split(train_size, val_size, test_size,
                          train_transform, val_test_transform, batch_size, labels) -> list[DataLoader]:
    val_sample = np.random.choice(np.arange(train_size), size=(val_size,), replace=False)
    val_indicator = np.zeros(train_size, dtype=bool)
    val_indicator[val_sample] = True
    train_ind = np.arange(train_size)[~val_indicator]
    val_ind = np.arange(train_size)[val_indicator]

    trainset = DLDataset(train_ind, test=False, transform=train_transform, labels=labels)
    valset = DLDataset(val_ind, test=False, transform=val_test_transform, labels=labels)
    testset = DLDataset(np.arange(test_size), test=True, transform=val_test_transform, labels=labels)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


n_classes = 200
out_channels = 32

class BasicBlockNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv2d_1 = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2d_res = nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=1)
        self.conv2d_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pooling = nn.AvgPool2d(8)

        self.linear = nn.Linear(in_features=800, out_features=n_classes)

    def forward(self, x):
        main_part = self.bn2(self.conv2d_2(self.relu(self.bn1(self.conv2d_1(x)))))
        resid_part = self.conv2d_res(x)
        block_out = self.pooling(self.relu(main_part + resid_part))
        out = self.linear(torch.flatten(block_out, start_dim=1))

        return out


from copy import copy

class MyGigaNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding='same', bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding='same', bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding='same', bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding='same', bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        for i in range(len(self.block1)):
            if isinstance(self.block1[i], nn.Conv2d):
                 nn.init.kaiming_normal_(self.block1[i].weight, mode='fan_out', nonlinearity='relu')

        self.block2 = nn.Sequential(
            nn.Conv2d(512, 64, 3, padding='same', bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding='same', bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding='same', bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, padding='same', bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        for i in range(len(self.block2)):
            if isinstance(self.block2[i], nn.Conv2d):
                 nn.init.kaiming_normal_(self.block2[i].weight, mode='fan_out', nonlinearity='relu')

        self.end = nn.Sequential(
            nn.Conv2d(3072, self.num_classes, 1, bias=False),
            nn.BatchNorm2d(self.num_classes),
            nn.AdaptiveAvgPool2d(1),
        )
        
        nn.init.kaiming_normal_(self.end[0].weight, mode='fan_out', nonlinearity='relu')

    def space_to_depth(self, x, block_size=2):
        n, c, h, w = x.size()
        unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
        return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)



    def forward(self, x):
        x = self.block1(x)
        skip_1 = x

        x = self.block2(x)

        skip_1 = self.space_to_depth(skip_1)
        x = torch.cat((x, skip_1), dim=1)

        x = self.end(x)
        x = x.view(x.size(0), -1)

        return x


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test(model, loader):
    loss_log = []
    acc_log = []
    model.eval()

    for data, target in tqdm(loader):
        # <your code here>
        data = data.to(device)
        target = target.flatten().long().to(device)

        with torch.no_grad():
          logits = model(data)
          loss = nn.functional.cross_entropy(logits, target)

        loss_log.append(loss.item())

        # <your code here>
        acc = (logits.argmax(dim=1) == target).sum() / len(target)

        acc_log.append(acc.item())

    return np.mean(loss_log), np.mean(acc_log)


def train_epoch(model, optimizer, train_loader):
    loss_log = []
    acc_log = []
    model.train()

    for data, target in tqdm(train_loader):
        # <your code here>
        data = data.to(device)
        target = target.flatten().to(device)

        optimizer.zero_grad()
        logits = model(data)
        loss = nn.functional.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()

        loss_log.append(loss.item())

        # <your code here>
        acc = (logits.argmax(dim=1) == target).sum() / len(target)

        acc_log.append(acc.item())

    return loss_log, acc_log


def train(model, optimizer, n_epochs, train_loader, val_loader, scheduler=None, test_loader=None, test_size=None):
    train_loss_log, train_acc_log, val_loss_log, val_acc_log = [], [], [], []

    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader)
        val_loss, val_acc = test(model, val_loader)

        train_loss_log.append(np.mean(train_loss))
        train_acc_log.append(np.mean(train_acc))

        val_loss_log.append(val_loss)
        val_acc_log.append(val_acc)

        print(f"Epoch {epoch}")
        print(f" train loss: {np.mean(train_loss)}, train acc: {np.mean(train_acc)}")
        print(f" val loss: {val_loss}, val acc: {val_acc}\n")

        if scheduler is not None:
            scheduler.step(np.mean(train_acc))
        if test_loader is not None and epoch % 3 == 2:
            print("SAVING TEST LABELS")
            # save_results(model, test_loader, test_size, filename='labels_test')

    return train_loss_log, train_acc_log, val_loss_log, val_acc_log


def save_results(model, test_loader, test_size, filename):
    test_labels = torch.Tensor(0).to(device)
    for data, _ in test_loader:
        data = data.to(device)
        with torch.no_grad():
          logits = model(data)

        pred = logits.argmax(dim=1)
        test_labels = torch.cat((test_labels, pred))

    picture_ids = ["test_" + f"{str(i).zfill(5)}" + ".jpg" for i in np.arange(test_size)]
    ans = pd.DataFrame({"Id": picture_ids, "Category": test_labels.cpu()})
    ans.to_csv(f"{filename}.csv", index=False)


def create_next_file_with_data(directory, data):
    files = os.listdir(directory)
    
    max_number = -1
    for file_name in files:
        try:
            number = int(file_name)
            if number > max_number:
                max_number = number
        except ValueError:
            continue
    
    next_number = max_number + 1
    new_file_name = str(next_number)
    new_file_path = os.path.join(directory, new_file_name)
    
    with open(new_file_path, 'w') as new_file:
        new_file.write(data)
    print(f"Logged in: {os.path.join(directory, new_file_name)}")


def main():
    print("Device ", device)

    config = {
        'n_epochs': 10,
        'batch_size': 128,
        'name': 'resnet34_aug'
    }

    labels = pd.read_csv("bhw1/labels.csv")
    train_size = 100000
    test_size = 10000
    val_size = 5000
    batch_size = config['batch_size']
    n_epochs = config['n_epochs']

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        # transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(3)]), p=0.2),
        # transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation((-15, 15))]), p=0.2),
        # transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=[0, 1], contrast=[0, 0.5])]), p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )

    net = resnet34()
    net = net.to(device)

    # lr_warmup_epochs = 5
    # lr_warmup_decay = 0.01
    # optimizer = optim.SGD(net.parameters(), lr=0.3, momentum=0.9, nesterov=True)
    # warmpup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=lr_warmup_decay, total_iters=lr_warmup_epochs)
    # main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs-lr_warmup_epochs, eta_min=0)

    # scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmpup_scheduler, main_scheduler], milestones=[lr_warmup_epochs], verbose=True)

    optimizer = optim.Adam(net.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)

    train_loader, val_loader, test_loader = my_train_val_test_split(train_size, val_size, test_size, transform_train, transform_test, batch_size, labels)
    train_loss_log, train_acc_log, val_loss_log, val_acc_log = train(net, optimizer, n_epochs, train_loader, val_loader, scheduler, test_loader, test_size)

    test_loss, test_acc = test(net, val_loader)
    print("VAL ACC: ", np.mean(test_acc))

    create_next_file_with_data('fuck_logs', json.dumps(
        dict(
            config=config,
            train_loss=train_loss_log,
            val_loss=val_loss_log,
            train_acc=train_acc_log,
            val_acc=val_acc_log,
        )
    ))

    save_results(net, test_loader, test_size, filename='labels_test')



if __name__ == '__main__':
    main()
