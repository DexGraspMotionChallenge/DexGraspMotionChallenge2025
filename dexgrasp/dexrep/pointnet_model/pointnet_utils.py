import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # self.ln1 = nn.LayerNorm([64, 2048])
        # self.ln2 = nn.LayerNorm([128, 2048])
        # self.ln3 = nn.LayerNorm([1024, 2048])
        # self.ln4 = nn.LayerNorm(512)
        # self.ln5 = nn.LayerNorm(256)
        self.iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1,9).cuda()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)


        if x.device != self.iden.device:
            self.iden = self.iden.to(x.device)
        x = x + self.iden.repeat(batchsize, 1)
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # self.ln1 = nn.LayerNorm([64, 2048])
        # self.ln2 = nn.LayerNorm([128, 2048])
        # self.ln3 = nn.LayerNorm([1024, 2048])
        # self.ln4 = nn.LayerNorm(512)
        # self.ln5 = nn.LayerNorm(256)

        self.k = k
        self.iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).cuda()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
        #     batchsize, 1)
        # if x.is_cuda:
        #     iden = iden.cuda()
        # a = self.iden.repeat(batchsize, 1)
        if x.device != self.iden.device:
            self.iden = self.iden.to(x.device)
        x = x + self.iden.repeat(batchsize, 1)
        x = x.view(-1, self.k, self.k)
        return x



class PointNetEncoder(nn.Module):
    def  __init__(self, global_feat=True, feature_transform=False, inchan=3, outchan=1024, norms='bn'):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(inchan)
        self.conv1 = torch.nn.Conv1d(inchan, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 256, 1)
        self.conv5 = torch.nn.Conv1d(256, outchan, 1)

        if norms=='bn':
            self.n1 = nn.BatchNorm1d(64)
            self.n2 = nn.BatchNorm1d(128)
            self.n3 = nn.BatchNorm1d(128)
            self.n4 = nn.BatchNorm1d(256)
            self.n5 = nn.BatchNorm1d(outchan)

        else:
            self.n1 = nn.Identity()
            self.n2 = nn.Identity()
            self.n3 = nn.Identity()
            self.n4 = nn.Identity()
            self.n5 = nn.Identity()

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.outchan = outchan
        if self.feature_transform:
            self.fstn = STNkd(k=64)


    def forward(self, x):
        B, D, N = x.size() #D=3+c, N=2048
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.n1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.n2(self.conv2(x)))
        x = F.relu(self.n3(self.conv3(x)))

        # ---- add extra layers ----
        x = F.relu(self.n4(self.conv4(x)))
        x = self.n5(self.conv5(x))

        x = torch.max(x, 2, keepdim=True)[0]
        globfeat = x.view(-1, self.outchan)
        if self.global_feat:
            x = globfeat.view(-1, self.outchan, 1).repeat(1, 1, N)
            return globfeat, pointfeat
        else:
            return globfeat, pointfeat #trans, trans_feat


class PointNetEncoder_sample(nn.Module):
    def  __init__(self, global_feat=True, feature_transform=False, inchan=3, outchan=1024, norms='bn'):
        super(PointNetEncoder_sample, self).__init__()
        self.stn = STN3d(inchan)
        self.conv1 = torch.nn.Conv1d(inchan, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 256, 1)
        self.conv5 = torch.nn.Conv1d(256, outchan, 1)

        if norms=='bn':
            self.n1 = nn.BatchNorm1d(64)
            self.n2 = nn.BatchNorm1d(128)
            self.n3 = nn.BatchNorm1d(128)
            self.n4 = nn.BatchNorm1d(256)
            self.n5 = nn.BatchNorm1d(outchan)

        else:
            self.n1 = nn.Identity()
            self.n2 = nn.Identity()
            self.n3 = nn.Identity()
            self.n4 = nn.Identity()
            self.n5 = nn.Identity()

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.outchan = outchan
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B_o = x.shape[0]
        if B_o > 100:
            x = self.sample(x)
        B, D, N = x.size() #D=3+c, N=2048
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.n1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.n2(self.conv2(x)))
        x = F.relu(self.n3(self.conv3(x)))

        # ---- add extra layers ----
        x = F.relu(self.n4(self.conv4(x)))
        x = self.n5(self.conv5(x))


        x = torch.max(x, 2, keepdim=True)[0]
        globfeat = x.view(-1, self.outchan)
        if B_o > 100:
            globfeat = torch.unsqueeze(globfeat, dim=1).repeat(1, 10, 1)
            pointfeat = torch.unsqueeze(pointfeat, dim=1).repeat(1, 10, 1, 1)
            globfeat = globfeat.view(-1, self.outchan)[:B_o]
            pointfeat = pointfeat.view(-1, pointfeat.shape[2], pointfeat.shape[3])[:B_o]
        if self.global_feat:
            x = globfeat.view(-1, self.outchan, 1).repeat(1, 1, N)
            return globfeat, pointfeat
        else:
            return globfeat, pointfeat #trans, trans_feat

    def sample(self, x, freq=10):
        return x[::freq]
def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
