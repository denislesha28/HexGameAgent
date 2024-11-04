import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from GameMeta import GameMeta


class BoardData(Dataset):
    """
    Data Class used for Training and Unwrapping the Datasets
    """
    def __init__(self, dataset):
        self.X = dataset[:, 0]
        self.y_p, self.y_v = dataset[:, 1], dataset[:, 2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_p[idx], self.y_v[idx]


class AlphaNet(nn.Module):
    def __init__(self, training: bool = False):
        super(AlphaNet, self).__init__()

        self.training = training

        self.conv1 = nn.Conv2d(in_channels=GameMeta.NN_INPUT_CHANNELS, out_channels=GameMeta.NN_NUM_CHANNELS, kernel_size=GameMeta.NN_KERNEL_SIZE, stride=GameMeta.NN_STRIDE_SIZE, padding=GameMeta.NN_PADDING_SIZE)
        self.conv2 = nn.Conv2d(in_channels=GameMeta.NN_NUM_CHANNELS, out_channels=GameMeta.NN_NUM_CHANNELS, kernel_size=GameMeta.NN_KERNEL_SIZE, stride=GameMeta.NN_STRIDE_SIZE, padding=GameMeta.NN_PADDING_SIZE)
        self.conv3 = nn.Conv2d(in_channels=GameMeta.NN_NUM_CHANNELS, out_channels=GameMeta.NN_NUM_CHANNELS, kernel_size=GameMeta.NN_KERNEL_SIZE, stride=GameMeta.NN_STRIDE_SIZE)
        self.conv4 = nn.Conv2d(in_channels=GameMeta.NN_NUM_CHANNELS, out_channels=GameMeta.NN_NUM_CHANNELS, kernel_size=GameMeta.NN_KERNEL_SIZE, stride=GameMeta.NN_STRIDE_SIZE)

        self.bn1 = nn.BatchNorm2d(GameMeta.NN_NUM_CHANNELS)
        self.bn2 = nn.BatchNorm2d(GameMeta.NN_NUM_CHANNELS)
        self.bn3 = nn.BatchNorm2d(GameMeta.NN_NUM_CHANNELS)
        self.bn4 = nn.BatchNorm2d(GameMeta.NN_NUM_CHANNELS)

        self.fc1 = nn.Linear(GameMeta.NN_NUM_CHANNELS * (GameMeta.NN_BOARD_SIZE - 4) * (GameMeta.NN_BOARD_SIZE - 4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, GameMeta.NN_ACTION_SIZE)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        s = s.view(-1, 1, GameMeta.NN_BOARD_SIZE, GameMeta.NN_BOARD_SIZE)  # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))  # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))  # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, GameMeta.NN_NUM_CHANNELS * (GameMeta.NN_BOARD_SIZE - 4) * (GameMeta.NN_BOARD_SIZE - 4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=GameMeta.NN_DROPOUT, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=GameMeta.NN_DROPOUT, training=self.training)  # batch_size x 512

        pi = self.fc3(s)  # batch_size x action_size
        v = self.fc4(s)  # batch_size x 1

        return F.softmax(pi, dim=1), torch.tanh(v)


class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, policy_pred, policy_target, value_pred, value_target):
        policy_loss = nn.CrossEntropyLoss()
        policy_loss_out = policy_loss(policy_pred, policy_target)
        value_loss = nn.MSELoss()(value_pred, value_target)
        return policy_loss_out + value_loss
