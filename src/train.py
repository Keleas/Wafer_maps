import os
import time
import argparse
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from sklearn.metrics import confusion_matrix, accuracy_score

import src.models as models
from src.logger import Logger
from src.create_data import TrainingDatabaseCreator, WaferDataset
from src.torchutils import EarlyStopping, AdamW, CyclicLRWithRestarts


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
        for inputs, label in tqdm(self.val_loader, ascii=True, desc='validation'):
            inputs, label = inputs.to(device), label.to(device)
            with torch.set_grad_enabled(False):
                out = self.model(inputs)
                loss = nn.CrossEntropyLoss()(out, label)

            _, y_pred = torch.max(out, 1)
            predicts.append(y_pred.cpu().numpy())
            truths.append(label.cpu().numpy())
            cum_loss += loss.item() * inputs.size(0)

        predicts = np.concatenate(predicts).squeeze()
        truths = np.concatenate(truths).squeeze()
        accuracy = (truths == predicts).mean()
        val_loss = cum_loss / self.val_data.__len__()
        return val_loss, accuracy

    def train_step(self):
        """

        :return:
        """
        cum_loss = 0

        self.model.train()
        for inputs, label in tqdm(self.train_loader, ascii=True, desc='train'):
            inputs, label = inputs.to(device), label.to(device)
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                out = self.model(inputs)
                loss = nn.CrossEntropyLoss()(out, label)

                loss.backward()
                self.optimizer.step()

            cum_loss += loss.item() * inputs.size(0)

        epoch_loss = cum_loss / self.train_data.__len__()
        return epoch_loss

    def logger_step(self, cur_epoch, losses_train, losses_val, accuracy):
        print(f"[Epoch {cur_epoch}] training loss={losses_train[-1]:.6f};  val_loss={losses_val[-1]:.6f}; "
              f"val_acc={accuracy:.6f}")
        print(f"Learning rate: {self.lr_scheduler.get_lr()[0]:.6f}")

        # 1. Log scalar values (scalar summary)
        info = {'loss': losses_train[-1],
                'val_loss': losses_val[-1],
                'val_acc': accuracy.item()}

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, cur_epoch + 1)

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in self.model.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary(tag, value.data.cpu().numpy(), cur_epoch + 1)
            self.logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), cur_epoch + 1)

        # 3. Log training images (image summary)
        self.model.eval()
        inputs, labels = next(iter(self.val_loader))
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.set_grad_enabled(False):
            out = self.model(inputs)

        _, y_pred = torch.max(out, 1)
        info = {'images': (inputs.view(-1, 96, 96)[:10].cpu().numpy(), y_pred.data.tolist(), labels.data.tolist())}

        for tag, images in info.items():
            self.logger.image_summary(tag, images, cur_epoch + 1)

    def start_train(self):
        """

        :return:
        """
        scheduler_step = args.epoch // args.snapshot

        num_snapshot = 0
        best_acc = 0

        # Setup optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.max_lr, momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, scheduler_step, args.min_lr)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        # Service variables
        losses_train = []  # save training losses
        losses_val = []  # save validation losses

        for epoch in range(args.epoch):
            train_loss = self.train_step()
            val_loss, accuracy = self.val_step()
            self.lr_scheduler.step()

            losses_train.append(train_loss)
            losses_val.append(val_loss)

            self.logger_step(epoch, losses_train, losses_val, accuracy)

            # scheduler checkpoint
            if accuracy > best_acc:
                best_acc = accuracy
                best_param = self.model.state_dict()

            if (epoch + 1) % scheduler_step == 0:
                torch.save(best_param, args.save_weight + args.weight_name + str(num_snapshot) + '.pth')
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.max_lr, momentum=args.momentum,
                                                 weight_decay=args.weight_decay)
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, scheduler_step,
                                                                               args.min_lr)
                num_snapshot += 1
                best_acc = 0

            # If the model could not get better, then stop training
            early_stopping(losses_val[-1], self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                # print(f"Early stop at {epoch} epoch")
                return

    def load_data(self):
        args_make_data = {'synthesized_path_name': args.synth_name,
                          'failure_types_ratio': {'Center': args.center_rate,
                                                  'Donut': args.donut_rate,
                                                  'Edge-Loc': args.edge_loc_rate,
                                                  'Edge-Ring': args.edge_ring_rate,
                                                  'Loc': args.loc_rate,
                                                  'Scratch': args.scratch_rate}
                          }
        data = TrainingDatabaseCreator()
        train, test, val = data.make_training_database(**args_make_data)

        print(np.unique(train.failureNum.values))

        train_data = WaferDataset(list(train.waferMap.values),
                                  mode='train', label_list=list(train.failureNum.values),
                                  fine_size=args.fine_size, pad_left=args.pad_left,
                                  pad_right=args.pad_right
                                  )
        self.train_data = train_data
        self.train_loader = DataLoader(train_data,
                                       shuffle=RandomSampler(train_data),
                                       batch_size=args.batch_size,
                                       num_workers=6,
                                       pin_memory=True)

        val_data = WaferDataset(list(val.waferMap.values),
                                mode='val', label_list=list(val.failureNum.values),
                                fine_size=args.fine_size, pad_left=args.pad_left,
                                pad_right=args.pad_right
                                )
        self.val_data = val_data
        self.val_loader = DataLoader(val_data,
                                     shuffle=RandomSampler(train_data),
                                     batch_size=args.batch_size,
                                     num_workers=6,
                                     pin_memory=True)

        test_data = WaferDataset(list(test.waferMap.values),
                                 mode='val', label_list=list(test.failureNum.values),
                                 fine_size=args.fine_size, pad_left=args.pad_left,
                                 pad_right=args.pad_right
                                 )
        self.test_data = test_data
        self.test_loader = DataLoader(test_data,
                                      shuffle=RandomSampler(train_data),
                                      batch_size=args.batch_size,
                                      num_workers=6,
                                      pin_memory=True)

        return True

    def plot_errors(self, model_name=None):
        print('[INFO] Plot errors...')
        cum_loss = 0
        predicts = []
        truths = []
        if model_name:
            checkpoint = torch.load(args.save_weight + model_name + '.pth')
        else:
            checkpoint = torch.load(args.save_weight + 'checkpoint.pth')
        test_model = self.model
        test_model.load_state_dict(checkpoint)
        test_model = test_model.to(device)

        self.model.eval()
        for inputs, label in tqdm(self.test_loader, ascii=True, desc='test'):
            inputs, label = inputs.to(device), label.to(device)
            with torch.set_grad_enabled(False):
                out = test_model(inputs)
                loss = nn.CrossEntropyLoss()(out, label)

            _, y_pred = torch.max(out, 1)
            predicts.append(y_pred.cpu().numpy())
            truths.append(label.cpu().numpy())
            cum_loss += loss.item() * inputs.size(0)

        predicts = np.concatenate(predicts).squeeze()
        truths = np.concatenate(truths).squeeze()
        accuracy = (truths == predicts).mean()
        val_loss = cum_loss / self.val_data.__len__()

        print(f"Top-1 Accuracy: {accuracy:.6f}, Loss: {val_loss:.6f}")

        def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

        cnf_matrix = confusion_matrix(truths, predicts)
        np.set_printoptions(precision=2)

        _, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
        types = ['Center', 'Donut', 'Edge-Loc',
                 'Edge-Ring', 'Loc', 'Scratch']
        l = np.arange(len(types))
        for ax in axes:
            ax.set_yticks(l)
            ax.set_yticklabels(types)
            ax.set_xticks(l)
            ax.set_xticklabels(types, rotation=45)

        # Plot non-normalized confusion matrix
        plt.subplot(axes[0])
        plot_confusion_matrix(cnf_matrix, title='Confusion matrix')

        # Plot normalized confusion matrix
        plt.subplot(axes[1])
        plot_confusion_matrix(cnf_matrix, normalize=True, title='Normalized confusion matrix')

        if not os.path.isdir('output/confusion_matrix/'):
            os.mkdir('output/confusion_matrix/')

        if model_name:
            plt.savefig('output/confusion_matrix/' + model_name + '.png')
        else:
            plt.savefig('output/confusion_matrix/' + args.weight_name + '.png')
        plt.show()

        return True

    def main(self):
        # Get Model
        self.model = models.ResNet(18)
        self.model.to(device)

        # Get Data
        self.load_data()  # train/val/test loaders

        # Start train loop
        # self.start_train()

        # Get train results
        # self.plot_errors(model_name='model_96_ResNet1710')

        return True


parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', default='ResNet18_v30000', type=str, help='Model version')
parser.add_argument('--fine_size', default=96, type=int, help='Resized image size')
parser.add_argument('--pad_left', default=0, type=int, help='Left padding size')
parser.add_argument('--pad_right', default=0, type=int, help='Right padding size')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--epoch', default=100, type=int, help='Number of training epochs')
parser.add_argument('--snapshot', default=20, type=int, help='Number of snapshots per fold')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--save_weight', default='output/weights/', type=str, help='weight save space')
parser.add_argument('--max_lr', default=0.01, type=float, help='max learning rate')
parser.add_argument('--min_lr', default=0.001, type=float, help='min learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--patience', default=40, type=int, help='Number of epoch waiting for best score')

parser.add_argument('--synth_name', default='synthesized_database_30000_v1.pkl', type=str, help='Synthesized path name')
parser.add_argument('--center_rate', default=0.1, type=float, help='Center rate of real data')
parser.add_argument('--donut_rate', default=0.1, type=float, help='Center rate of real data')
parser.add_argument('--edge_loc_rate', default=0.1, type=float, help='Edge-Loc rate of real data')
parser.add_argument('--edge_ring_rate', default=0.1, type=float, help='Edge-Ring rate of real data')
parser.add_argument('--loc_rate', default=0.1, type=float, help='Loc rate of real data')
parser.add_argument('--scratch_rate', default=0.1, type=float, help='Scratch rate of real data')

args = parser.parse_args()
fine_size = args.fine_size + args.pad_left + args.pad_right
args.weight_name = 'model_' + str(fine_size) + '_' + args.model

if not os.path.isdir(args.save_weight):
    os.mkdir(args.save_weight)

device = torch.device('cuda' if args.cuda else 'cpu')


if __name__ == '__main__':
    # ================================================================== #
    #                        Tensorboard Logging                         #
    # ================================================================== #
    # tensorboard --logdir=D:\Github\Wafer_maps\logs --port=6006
    # http://localhost:6006/

    model = TrainModel()
    model.main()

