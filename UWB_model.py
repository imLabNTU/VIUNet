import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from torch.autograd import Variable
from torchvision import transforms
import math

class UWBDataset(Dataset):
    def __init__(self, data_dir, idx, transform=None, target_transform=None):
        
        self.data = []

        self.transform = transforms.Compose([transforms.ToTensor()])

        for i in idx:
            data = pd.read_csv(data_dir / f"{i:02d}" / "uwbs.txt", header=None).to_numpy().astype(np.float32)
            uwb = data[:, :16]
            gt_data = pd.read_csv(data_dir / f"{i:02d}" / "poses.txt", header=None, sep = " |,", engine="python").to_numpy().astype(np.float32)
            gt_x = gt_data[:,3]
            gt_y = gt_data[:,7]
            gt_z = gt_data[:,11]
            gt = np.stack((gt_x, gt_y, gt_z), axis = 1)
            # gt = data[:, 16:]

            if len(self.data) == 0:
                self.data = np.concatenate((uwb, gt), axis = 1)
            else:
                data = np.concatenate((uwb, gt), axis = 1)
                self.data = np.concatenate((self.data, data), axis = 0)

        self.data = self.data.astype(np.float32)
        # print(self.data.shape)

        # uwb.to_numpy()[4:8, :]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(np.expand_dims(self.data[idx][:16], 0)), self.transform(np.expand_dims(self.data[idx][16:], 0))


class Net(nn.Module):
    def __init__(self, indim, hiddim, outdim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(indim, hiddim)
        self.fc2 = nn.Linear(hiddim, hiddim)
        self.fc3 = nn.Linear(hiddim, outdim)
        self.dropout = nn.Dropout(0)

    def forward(self, x):
        # Pass data through fc1
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        return x

def main():

    data_dir = Path("../select_fusion/euroc_new/")
    train_set = UWBDataset(data_dir, [0, 1,10,3,4,6,7,8,9])
    test_set = UWBDataset(data_dir, [5])
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=10, shuffle=False)

    model = Net(16, 256, 3).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0)

    n_epoch = 100
    best_error = 1000000

    mse = torch.nn.MSELoss()

    for epoch in range(n_epoch):

        model.train()
        true_loss, total_loss, n = 0, 0, 0
        batch_time = datetime.now().timestamp()

        for i, (e, gt) in enumerate(train_loader):

            # print(e)
            # print(gt)
            e = Variable(e.cuda().squeeze())
            gt = Variable(gt.cuda().squeeze())
            est = model(e)
            # print(est.shape)

            loss = mse(est, gt) #+ mse(est[:,2], gt[:,2])
            for j in range(len(est)):
                total_loss += math.sqrt((est[j] - gt[j]).pow(2).sum().item())
                n += 1
            true_loss += loss.item()
            # total_loss += loss.item()

            if i == 0 and epoch == 0:
                print("==============================")
                print(est[0])
                print(gt[0])
                print(est[1])
                print(gt[1])
                print("==============================")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # total_loss /= len(train_loader)

        print('Train: Epoch [{}/{}] Time {:.4}s Loss {} / {}'.
                format(epoch + 1, n_epoch, datetime.now().timestamp() - batch_time, total_loss/n, true_loss))

        model.eval()

        total_loss, n = 0, 0

        for i, (e, gt) in enumerate(test_loader):
            e = Variable(e.cuda().squeeze())
            gt = Variable(gt.cuda().squeeze())
            est = model(e)
            for j in range(len(est)):
                total_loss += math.sqrt((est[j] - gt[j]).pow(2).sum().item())
                n += 1
            # loss = mse(est/1000, gt/1000)
            # total_loss += loss.item() * 3
            if i == 0 and epoch == 0:
                print("--------------------------------")
                print(est[0])
                print(gt[0])
                print(est[1])
                print(gt[1])
                print("--------------------------------")
        
        # print(total_loss/n)
        print('Test: Epoch [{}/{}] mean error: {}'.
                format(epoch + 1, n_epoch, total_loss/n))

        if total_loss/n < best_error:
            best_error = total_loss/n
            fn = Path("./pretrain/uwbtest/") / "uwb_encoder_euroc_10cm.pth"
            torch.save(model.state_dict(), str(fn))

if __name__ == '__main__':
    main()
