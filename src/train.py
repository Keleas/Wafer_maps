import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from IPython.display import clear_output
import time
from sklearn.metrics import confusion_matrix, accuracy_score
import hiddenlayer as hl
from sklearn.preprocessing import MultiLabelBinarizer
import itertools

import src.models as models
import src.create_data as create_data
from src.logger import Logger


class TrainModel(object):
    def __init__(self):
        self.is_cuda = torch.cuda.is_available()
        self.model = models.BN_LeNet()
        if self.is_cuda:
            self.model = self.model.to("cuda:0")
        self.image_reshape = (-1, 1, 96, 96)
        self.logger = Logger('logs/')

        self.train = None
        self.test = None
        self.val = None
        self.mlb = None

    def batch_generator(self, batch_size=10, mode='train'):
        if mode == 'train':
            data = self.train
        elif mode == 'test':
            data = self.test
        elif mode == 'val':
            data = self.val
        else:
            raise Exception('Invalid batch mode')
        out_map = []
        out_class = []
        while True:
            data = data.sample(frac=1).reset_index(drop=True)
            for index, row in data.iterrows():
                out_map += [row.waferMap]
                out_class += [row.failureNum]
                if len(out_map) >= batch_size:
                    out_map = np.array(out_map)
                    out_map = out_map.reshape(self.image_reshape)
                    # out_class = self.mlb.transform(out_class)
                    if self.is_cuda:
                        yield (torch.tensor(out_map, dtype=torch.float32).cuda(),
                               torch.LongTensor(out_class).cuda())
                    else:
                        yield (torch.tensor(out_map, dtype=torch.float32),
                               torch.LongTensor(out_class))
                    out_map = []
                    out_class = []

    def start_train_model(self, num_epo=10, batch_size=10,
              weights_file="model.torch", checkpoint_after=10,
              lr_start=1e-3, lr_deacy_rate=10, drop_lr_after=10):
        """
        Traininig the given model
        """
        opt = torch.optim.Adam(self.model.parameters(), lr=lr_start)  # Define the optimizer
        loss_f = nn.CrossEntropyLoss()  # Define loss function used

        # Service variables
        losses_train = []  # save training losses
        losses_val = []  # save validation loasses
        cur_step = 0  # current iteration
        steps_per_epoch = 100
        steps_per_epoch_val = 100

        start = time.time()
        self.model = self.model.train()
        for epo in range(num_epo):
            cum_loss = 0  # Zero the cumulative sum
            for step in range(steps_per_epoch):
                x, y = next(self.batch_generator(batch_size=batch_size, mode='train'))
                out = self.model(x)  # calculate model outputs
                loss = loss_f(out, y)  # calculate loss

                cum_loss += loss.data.cpu().item()  # add current loss to cum_loss

                opt.zero_grad()  # zero old gradients
                loss.backward()  # calculate new gradients
                opt.step()  # perform optimization step

                cur_step += 1
                if cur_step % checkpoint_after == 0:
                    val_loss = 0
                    # measure loss on validation
                    for val_step in range(steps_per_epoch_val):
                        x, y = next(self.batch_generator(batch_size=batch_size, mode='val'))
                        out = self.model(x)
                        val_loss += loss_f(out, y).data.cpu().item()
                    # add new data to lists
                    losses_train.append(cum_loss / checkpoint_after)
                    losses_val.append(val_loss / checkpoint_after)
                    cum_loss = 0

                    if np.argmin(losses_val) == len(losses_val) - 1:  # if new loss is the best
                        torch.save(self.model, weights_file)  # save model

                    # if there was no improvement for drop_lr_after iterations
                    if len(losses_val) - 1 - np.argmin(losses_val) > drop_lr_after:
                        model = torch.load(weights_file)  # load the best model
                        lr_start /= lr_deacy_rate  # reduce learning rate
                        opt = torch.optim.Adam(model.parameters(), lr=lr_start)

                    # plot losses
                    clear_output(True)
                    # Compute accuracy
                    _, y_pred = torch.max(out, 1)
                    accuracy = (y == y_pred.squeeze()).float().mean()
                    print(f"Epoch {epo}: trining loss={losses_train[-1]}; test loss={losses_val[-1]}; "
                          f"val_acc={accuracy}")
                    print(f"Learninig rate: {lr_start}")
                    end = time.time()
                    print(f"Iteration took {end - start}s.")

                    start = end
                    # plt.plot(losses_train, c="r")
                    # plt.plot(losses_val, c="g")
                    # plt.show()

                    # ================================================================== #
                    #                        Tensorboard Logging                         #
                    # ================================================================== #

                    # tensorboard --logdir=D:\Github\Wafer_maps\logs --port=6006
                    # http://localhost:6006/

                    # 1. Log scalar values (scalar summary)
                    info = {'loss': loss.item(), 'accuracy': accuracy.item()}

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

                    # If the model could not get better even after lr decay, then stop training
                    if lr_start < 1e-6:
                        print(f"Early stop at {cur_step}")
                        return

    def plot_architecture(self):
        print(self.model)
        w = torch.zeros([2, 1, 96, 96])
        if self.is_cuda:
            w = w.cuda()
            self.model = self.model.to("cuda:0")

        return hl.build_graph(self.model, w)

    def load_data(self):
        args = {'synthesized_path_name': 'synthesized_test_database.pkl',
                'failure_types_ratio': {'Center': 0.1,
                                        'Donut': 0.1,
                                        'Edge-Loc': 0.1,
                                        'Edge-Ring': 0.1,
                                        'Loc': 0.1,
                                        'Random': 0.1,
                                        'Scratch': 0.1}
                }
        data = create_data.TrainingDatabaseCreator()
        self.train, self.test, self.val = data.make_training_database(**args)

        labels = [None] * self.train.shape[0]

        i = 0
        for index, row in self.train.iterrows():
            label = row.failureType[0]
            labels[i] = label
            i += 1

        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(labels)
        self.mlb = mlb

        # print(labels.shape)
        # print("class labels:")
        # for (i, label) in enumerate(mlb.classes_):
        #     print("{}. {}".format(i + 1, label))

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
                out_map = out_map.reshape(self.image_reshape)
                if self.is_cuda:
                    x = torch.tensor(out_map, dtype=torch.float32).cuda()
                    y = torch.LongTensor(out_class).cuda()
                else:
                    x = torch.tensor(out_map, dtype=torch.float32)
                    y = torch.LongTensor(out_class)

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
        self.load_data()
        if self.is_cuda:
            self.model.to("cuda:0")

        self.start_train_model(num_epo=10, batch_size=100,
                               weights_file="output/models/lenet.torch", checkpoint_after=10,
                               lr_start=1e-3, lr_deacy_rate=10, drop_lr_after=10)

        self.plot_errors()

        return True


model = TrainModel()
model.main()

