import os
import argparse
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import gc

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
import segmentation_models_pytorch as smp
from sklearn.metrics import confusion_matrix
import torchvision.models as torch_models

from src.utils.logger import Logger
# from src.data_generation.create_data import TrainingDatabaseCreator, WaferDataset
from src.utils.torchutils import EarlyStopping
from src.utils.losses import lovasz_softmax, dice_channel_torch


class WaferDataset(Dataset):
    def __init__(self, wafer_list, mask_list):
        self.wafer_list = wafer_list
        self.mask_list = mask_list

    def __len__(self):
        return len(self.wafer_list)

    def __getitem__(self, idx):
        wafer_map = deepcopy(self.wafer_list[idx])
        pattern_mask = deepcopy(self.mask_list[idx])

        wafer_map, pattern_mask = torch.from_numpy(np.repeat(np.expand_dims(wafer_map, axis=0), 3, axis=0)).float(), \
                                  torch.from_numpy(pattern_mask).float()

        return wafer_map, pattern_mask


class TrainModel(object):
    def __init__(self):
        self.logger = Logger('logs/')

        # model
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None

        # data
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def val_step(self):
        """

        :return:
        """
        cum_loss = 0
        predicts = []
        truths = []

        self.model.eval()
        for wafer_map, pattern_mask in tqdm(self.val_loader, total=len(self.val_loader), ascii=True, desc='validation'):
            wafer_map, pattern_mask = wafer_map.to(device), pattern_mask.to(device)
            with torch.set_grad_enabled(False):
                out = self.model(wafer_map)
                #loss = nn.BCEWithLogitsLoss()(out, pattern_mask)
                loss = nn.BCEWithLogitsLoss()(F.sigmoid(out), pattern_mask)

            predicts.append(F.sigmoid(out).detach().cpu().numpy())
            truths.append(pattern_mask.detach().cpu().numpy())
            cum_loss += loss.item() * wafer_map.size(0)
            gc.collect()

        predicts = np.concatenate(predicts)
        truths = np.concatenate(truths)
        mean_dice = dice_channel_torch(predicts, truths, 0.5)
        val_loss = cum_loss / self.val_loader.__len__()
        return val_loss, mean_dice

    def train_step(self):
        cum_loss = 0

        self.model.train()
        for wafer_map, pattern_mask in tqdm(self.train_loader):
            wafer_map, pattern_mask = wafer_map.to(device), pattern_mask.to(device)
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                out = self.model(wafer_map)
                loss = nn.BCEWithLogitsLoss()(F.sigmoid(out), pattern_mask)

                loss.backward()
                self.optimizer.step()
                gc.collect()

            cum_loss += loss.item() * wafer_map.size(0)

        epoch_loss = cum_loss / self.train_loader.__len__()
        gc.collect()
        return epoch_loss

    def logger_step(self, cur_epoch, losses_train, losses_val, mean_dice):
        print(f"[Epoch {cur_epoch}] training loss={losses_train:.6f};  val_loss={losses_val:.6f}; "
              f"val_acc={mean_dice:.6f}")

        # 1. Log scalar values (scalar summary)
        info = {'loss': losses_train,
                'val_loss': losses_val,
                'val_acc': mean_dice}

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, cur_epoch + 1)

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in self.model.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary(tag, value.data.cpu().numpy(), cur_epoch + 1)
            self.logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), cur_epoch + 1)

    def training(self):

        num_snapshot = 0
        best_acc = 0

        # Service variables
        losses_train = []  # save training losses
        losses_val = []  # save validation losses

        for epoch in range(args.epoch):
            train_loss = self.train_step()
            val_loss, mean_dice = self.val_step()
            self.lr_scheduler.step(val_loss)

            losses_train.append(train_loss)
            losses_val.append(val_loss)

            self.logger_step(epoch, train_loss, val_loss, mean_dice)

            # scheduler checkpoint
            if mean_dice > best_acc:
                best_acc = mean_dice
                best_param = self.model.state_dict()
                torch.save(best_param, args.save_weight + args.weight_name + str(num_snapshot) + '.pth')

        return True

    def read_dataframe_in_torch(self, path_to_file):
        pd_database = pd.read_pickle(path_to_file)

        pd_database = pd_database.reset_index()
        pd_database['wafer_map'] = pd_database.wafer_map
        pd_database['pattern_mask'] = pd_database.pattern_mask

        ######################
        # Get fixed size of maps
        out_map = []
        out_mask = []
        dim_size = 40  # минимальный размер пластины(wafer)
        for index, row in pd_database.iterrows():
            waf_map = row.wafer_map
            waf_mask = row.pattern_mask

            if waf_map.shape[0] > dim_size and waf_map.shape[1] > dim_size:
                out_map += [waf_map]
                out_mask += [waf_mask[0]]
                # [0]-костыль TODO: заменить костыль

        pd_database = pd.DataFrame(data=np.vstack((out_map, out_mask)).T, columns=['wafer_map', 'pattern_mask'])

        pd_database.to_pickle(pd_database)

        return pd_database

    def load_data(self):
        '''
        Загружает данные в обучающую и валидационную выборки
        '''
        path_to_file = "/Users/user/K-132-8/Wafer_maps/input/synthesis/test_database.pkl"
        # database_pd = self.read_dataframe_in_torch(path_to_file)  # TODO: починить, если понадобится загрузка реальных
        database_pd = pd.read_pickle(path_to_file)
        data_set = WaferDataset(list(database_pd.wafer_map.values), mask_list=list(database_pd.pattern_mask.values))

        validation_split = .1
        shuffle_dataset = True
        random_seed = 42
        dataset_size = len(data_set)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        self.train_loader = torch.utils.data.DataLoader(data_set, sampler=train_sampler, batch_size=args.batch_size,
                                                        shuffle=False)
        self.val_loader = torch.utils.data.DataLoader(data_set, sampler=valid_sampler, batch_size=args.batch_size,
                                                      shuffle=False)

        return True

    def main(self):
        # Get Model
        self.model = smp.Unet(args.model, classes=1)
        self.model.to(device)
        print(self.model)

        # Get Data
        self.load_data()  # train/val/test loaders

        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.max_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=args.patience)
        self.training()
        return True


parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', default='resnet34', type=str, help='Model version')
parser.add_argument('--fine_size', default=96, type=int, help='Resized image size')
parser.add_argument('--pad_left', default=0, type=int, help='Left padding size')
parser.add_argument('--pad_right', default=0, type=int, help='Right padding size')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--epoch', default=50, type=int, help='Number of training epochs')
parser.add_argument('--snapshot', default=2, type=int, help='Number of snapshots per fold')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--save_weight', default='output/weights/', type=str, help='weight save space')
parser.add_argument('--max_lr', default=1e-3, type=float, help='max learning rate')
parser.add_argument('--min_lr', default=1e-5, type=float, help='min learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight decay for SGD')
parser.add_argument('--patience', default=40, type=int, help='Number of epoch waiting for best score')

parser.add_argument('--synth_name', default='test_database.csv', type=str, help='Synthesized path name')
parser.add_argument('--real_name', default='_', type=str, help='Real wafers path name')
parser.add_argument('--each_rate', default=120, type=float, help='Rate of real data')
parser.add_argument('--center_rate', default=0.1, type=float, help='Center rate of real data')
parser.add_argument('--donut_rate', default=0.1, type=float, help='Center rate of real data')
parser.add_argument('--edge_loc_rate', default=0.1, type=float, help='Edge-Loc rate of real data')
parser.add_argument('--edge_ring_rate', default=0.1, type=float, help='Edge-Ring rate of real data')
parser.add_argument('--loc_rate', default=0.1, type=float, help='Loc rate of real data')
parser.add_argument('--scratch_rate', default=0.1, type=float, help='Scratch rate of real data')
parser.add_argument('--none_rate', default=0.1, type=float, help='Random rate of real data')

args = parser.parse_args()
fine_size = args.fine_size + args.pad_left + args.pad_right
args.weight_name = 's' + str(fine_size) + '_r' + str(args.each_rate) + '_' +\
                   args.model + '_' + args.synth_name
args.weight_name = args.weight_name[:-4]

if not os.path.isdir(args.save_weight):
    os.mkdir(args.save_weight)

device = torch.device('cuda' if args.cuda else 'cpu')


if __name__ == '__main__':
    # ================================================================== #
    #                        Tensorboard Logging                         #
    # ================================================================== #
    # tensorboard --logdir=C:\Users\user\K-132-8\Wafer_maps\src\main\logs --port=6006
    # http://localhost:6006/
    model = TrainModel()
    model.main()

