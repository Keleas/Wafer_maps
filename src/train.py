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
        self.model = None
        self.loss = None
        self.optimizer = None
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def val_step(self):
        loss_f = nn.CrossEntropyLoss()
        pass

    def train_step(self):
        loss_f = nn.CrossEntropyLoss()
        data_size = self.train_data.__len__()

        cum_loss = 0  # Zero the cumulative sum
        for step in tqdm(range(steps_per_epoch)):
            x, y = next(self.batch_generator(batch_size=batch_size, mode='train'))
            out = self.model(x)  # calculate model outputs
            loss = loss_f(out, y)  # calculate loss

            cum_loss += loss.data.cpu().item()  # add current loss to cum_loss

            self.optimizer.zero_grad()  # zero old gradients
            loss.backward()  # calculate new gradients
            self.optimizer.step()  # perform optimization step

        pass

    def logger_step(self, losses_train, losses_val, accuracy):
        pass

    def start_train(self):
        scheduler_step = args.epoch // args.snapshot

        num_snapshot = 0
        best_acc = 0

        # Setup optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.max_lr, momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, scheduler_step, args.min_lr)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        # Service variables
        losses_train = []  # save training losses
        losses_val = []  # save validation losses

        for epoch in range(args.epoch):
            train_loss = self.train_step()
            val_loss, accuracy = self.val_step()
            lr_scheduler.step()

            losses_train.append(train_loss)
            losses_val.append(val_loss)

            self.logger_step(losses_train, losses_val, accuracy)

            # scheduler checkpoint
            if accuracy > best_acc:
                best_acc = accuracy
                best_param = self.model.state_dict()

            if (epoch + 1) % scheduler_step == 0:
                torch.save(best_param, args.save_weight + args.weight_name + str(num_snapshot) + '.pth')
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.max_lr, momentum=args.momentum,
                                                 weight_decay=args.weight_decay)
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, scheduler_step, args.min_lr)
                num_snapshot += 1
                best_acc = 0

            # If the model could not get better, then stop training
            early_stopping(losses_val[-1], self.model)

            if early_stopping.early_stop:
                print("Early stopping")
                print(f"Early stop at {epoch} epoch")
                return

    def start_train_model(self, num_epo=10, batch_size=10, steps_per_epoch=100, steps_per_epoch_val=100,
                          lr_start=1e-3, lr_weight_decay=1e-5, lr_restart_period=3,
                          lr_t_mult=1.2, lr_policy="cosine",
                          patience=20, weights_file="model.torch", checkpoint_after=10):
        """
        Traininig the given model
        """
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_start)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=0.01, max_lr=0.1)

        # optimizer = AdamW(self.model.parameters(), lr=lr_start, weight_decay=lr_weight_decay)
        # batch_size = batch_size
        # epoch_size = batch_size * steps_per_epoch
        # scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size,
        #                                  restart_period=lr_restart_period,
        #                                  t_mult=lr_t_mult,
        #                                  policy=lr_policy)

        scheduler_step = args.epoch // args.snapshot

        # Setup optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=args.max_lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step, args.min_lr)
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        loss_f = nn.CrossEntropyLoss()

        # Service variables
        losses_train = []  # save training losses
        losses_val = []  # save validation losses
        cur_step = 0  # current iteration

        start = time.time()
        for epo in range(num_epo):

            ###################
            # train the model #
            ###################
            self.model = self.model.train()
            cum_loss = 0  # Zero the cumulative sum
            checkpoint_after = steps_per_epoch
            if epo:
                lr_scheduler.step()
            for step in tqdm(range(steps_per_epoch)):
                x, y = next(self.batch_generator(batch_size=batch_size, mode='train'))
                out = self.model(x)  # calculate model outputs
                loss = loss_f(out, y)  # calculate loss

                cum_loss += loss.data.cpu().item()  # add current loss to cum_loss

                optimizer.zero_grad()  # zero old gradients
                loss.backward()  # calculate new gradients
                optimizer.step()  # perform optimization step

                cur_step += 1

            ######################
            # validate the model #
            ######################
            if cur_step % checkpoint_after == 0:
                self.model = self.model.eval()
                val_loss = 0
                # measure loss on validation
                for val_step in range(steps_per_epoch_val):
                    x, y = next(self.batch_generator(batch_size=batch_size, mode='val'))
                    out = self.model(x)
                    val_loss += loss_f(out, y).data.cpu().item()

                # add new data to lists
                losses_train.append(cum_loss / checkpoint_after)
                losses_val.append(val_loss / steps_per_epoch_val)

                if np.argmin(losses_val) == len(losses_val) - 1:  # if new loss is the best
                    torch.save(self.model, weights_file)  # save model

                # Compute accuracy
                _, y_pred = torch.max(out, 1)
                accuracy = (y == y_pred.squeeze()).float().mean()
                print('######################')
                print(f"[Epoch {epo}] training loss={losses_train[-1]:.6f};  val_loss={losses_val[-1]:.6f}; "
                      f"val_acc={accuracy:.6f}")
                # print(f"Learning rate: {list(scheduler.get_lr(scheduler.t_cur))}")
                end = time.time()
                print(f"Iteration took {end - start:.2f}s.")

                start = end
                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #

                # tensorboard --logdir=D:\Github\Wafer_maps\logs --port=6006
                # http://localhost:6006/

                # 1. Log scalar values (scalar summary)
                info = {'loss': losses_train[-1],
                        'val_loss': losses_val[-1],
                        'val_acc': accuracy.item()}

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, cur_step + 1)

                # 2. Log values and gradients of the parameters (histogram summary)
                for tag, value in self.model.named_parameters():
                    tag = tag.replace('.', '/')
                    self.logger.histo_summary(tag, value.data.cpu().numpy(), cur_step + 1)
                    self.logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), cur_step + 1)

                # 3. Log training images (image summary)
                info = {'images': (x.view(-1, 96, 96)[:10].cpu().numpy(), y_pred.data.tolist())}

                for tag, images in info.items():
                    self.logger.image_summary(tag, images, cur_step + 1)

                # If the model could not get better, then stop training
                early_stopping(losses_val[-1], self.model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    print(f"Early stop at {cur_step}")
                    return

    def load_data(self):
        args_make_data = {'synthesized_path_name': args.synth_name,
                          'failure_types_ratio': {'Center': args.center_rate,
                                                  'Donut': args.donut_rate,
                                                  'Edge-Loc': args.edge_loc.rate,
                                                  'Edge-Ring': args.edge_ring.rate,
                                                  'Loc': args.loc_rate,
                                                  'Random': args.random_rate,
                                                  'Scratch': args.scratch_rate}
                          }
        data = TrainingDatabaseCreator()
        train, test, val = data.make_training_database(**args_make_data)

        train_data = WaferDataset(list(train.waferMap.values),
                                  mode='train', label_list=list(train.failureNum.values),
                                  fine_size=args.fine_size, pad_left=args.pad_left,
                                  pad_right=args.pad_right
                                  )
        self.train_data = train_data
        self.train_loader = DataLoader(train_data,
                                       shuffle=RandomSampler(train_data),
                                       batch_size=50,
                                       num_workers=1,
                                       pin_memory=True)

        val_data = WaferDataset(list(val.waferMap.values),
                                mode='val', label_list=list(val.failureNum.values),
                                fine_size=args.fine_size, pad_left=args.pad_left,
                                pad_right=args.pad_right
                                )
        self.val_data = val_data
        self.val_loader = DataLoader(val_data,
                                     shuffle=RandomSampler(train_data),
                                     batch_size=50,
                                     num_workers=1,
                                     pin_memory=True)

        test_data = WaferDataset(list(test.waferMap.values),
                                 mode='val', label_list=list(test.failureNum.values),
                                 fine_size=args.fine_size, pad_left=args.pad_left,
                                 pad_right=args.pad_right
                                 )
        self.test_data = test_data
        self.test_loader = DataLoader(test_data,
                                      shuffle=RandomSampler(train_data),
                                      batch_size=50,
                                      num_workers=1,
                                      pin_memory=True)

        return True

    def plot_errors(self):
        print('[INFO] Plot errors...')
        self.model = self.model.eval()
        out_map = []
        out_class = []
        data = self.test.sample(frac=1).reset_index(drop=True)
        y_pred = []
        y_true = []
        for index, row in data.iterrows():
            out_map += [row.waferMap]
            out_class += [row.failureNum]

            if len(out_map) == 200:
                out_map = np.array(out_map)
                out_map = out_map.reshape((-1, 1, args.fine_size, args.fine))

                x = torch.tensor(out_map, dtype=torch.float32).to(device)
                y = torch.LongTensor(out_class).to(device)

                y_pred.append(torch.argmax(self.model(x), 1).cpu().numpy())
                y_true.append(y.cpu().numpy())

                out_map = []
                out_class = []

        y_pred = np.array(y_pred).flatten()
        y_true = np.array(y_true).flatten()
        print(f"Top-1 Accuracy: {accuracy_score(y_pred, y_true)}")

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

        cnf_matrix = confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)

        _, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
        types = ['Center', 'Donut', 'Edge-Loc',
                 'Edge-Ring', 'Loc', 'Random',
                 'Scratch']
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

        plt.show()

        return True

    def main(self):
        # Get Model
        self.model = models.ResNet(17)
        self.model.to(device)

        # Get Data
        self.load_data()  # train/val/test loaders

        # Start train loop
        self.start_train()

        self.start_train_model(num_epo=100, batch_size=10, steps_per_epoch=100, steps_per_epoch_val=100,
                               lr_start=1e-3, lr_weight_decay=1e-5, lr_restart_period=10,
                               lr_t_mult=1.2, lr_policy="cosine",
                               patience=30, weights_file="output/models/ResNet17.torch", checkpoint_after=10
                               )

        # Get train results
        self.plot_errors()

        return True


parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', default='res34v5', type=str, help='Model version')
parser.add_argument('--fine_size', default=101, type=int, help='Resized image size')
parser.add_argument('--pad_left', default=13, type=int, help='Left padding size')
parser.add_argument('--pad_right', default=14, type=int, help='Right padding size')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--epoch', default=300, type=int, help='Number of training epochs')
parser.add_argument('--snapshot', default=5, type=int, help='Number of snapshots per fold')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--save_weight', default='weights/', type=str, help='weight save space')
parser.add_argument('--max_lr', default=0.01, type=float, help='max learning rate')
parser.add_argument('--min_lr', default=0.001, type=float, help='min learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--patience', default=20, type=int, help='Number of epoch waiting for best score')

parser.add_argument('--synth_name', default='synthesized_test_database.pkl', type=str, help='Synthesized path name')
parser.add_argument('--center_rate', default=0.1, type=float, help='Center rate of real data')
parser.add_argument('--donut_rate', default=0.1, type=float, help='Center rate of real data')
parser.add_argument('--edge_loc_rate', default=0.1, type=float, help='Edge-Loc rate of real data')
parser.add_argument('--edge_ring_rate', default=0.1, type=float, help='Edge-Ring rate of real data')
parser.add_argument('--loc_rate', default=0.1, type=float, help='Loc rate of real data')
parser.add_argument('--random_rate', default=0.1, type=float, help='Random rate of real data')
parser.add_argument('--scratch_rate', default=0.1, type=float, help='Scratch rate of real data')

args = parser.parse_args()
fine_size = args.fine_size + args.pad_left + args.pad_right
args.weight_name = 'model_' + str(fine_size) + '_' + args.model

if not os.path.isdir(args.save_weight):
    os.mkdir(args.save_weight)

device = torch.device('cuda' if args.cuda else 'cpu')


if __name__ == '__main__':

    model = TrainModel()
    model.main()

