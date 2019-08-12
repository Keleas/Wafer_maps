import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import autograd as ag
from IPython.display import clear_output
import os
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix, accuracy_score
import hiddenlayer as hl

import src.models as models
import src.create_data as create_data

global is_cuda
is_cuda = torch.cuda.is_available()

def batch_generator(path, batch_size=10):
    for file_name in os.listdir(path):
        with open(path+file_name, 'rb') as fo:
            dict_ = pickle.load(fo, encoding='bytes')
        label = dict_[b"labels"]
        data = dict_[b"data"].reshape((-1,3,32,32))
        if is_cuda:
            for i in range(0,len(label),batch_size):
                yield (torch.tensor(data[i:i+batch_size], dtype=torch.float32).cuda(), torch.LongTensor(label[i:i+batch_size]).cuda())
        else:
            for i in range(0,len(label),batch_size):
                yield (torch.tensor(data[i:i+batch_size], dtype=torch.float32), torch.LongTensor(label[i:i+batch_size]))


def test_batch_generator(path, batch_size=10):
    with open(path + 'test_batch', 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    label = dict_[b"labels"]
    data = dict_[b"data"].reshape((-1,3,32,32))
    if is_cuda:
        for i in range(0,len(label),batch_size):
            yield (torch.tensor(data[i:i+batch_size], dtype=torch.float32).cuda(), torch.LongTensor(label[i:i+batch_size]).cuda())
    else:
        for i in range(0,len(label),batch_size):
            yield (torch.tensor(data[i:i+batch_size], dtype=torch.float32), torch.LongTensor(label[i:i+batch_size]))


def train(model, training_path, test_path, num_epo=10, batch_size=10,
          weights_file="model.torch", checkpoint_after=1000,
          lr_start=1e-3, lr_deacy_rate=10, drop_lr_after=10):
    """
    Traininig the given model
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr_start)  # Define the optimizer
    loss_f = nn.CrossEntropyLoss()  # Define loss function used

    # Service variables
    losses_train = []  # save training losses
    losses_val = []  # save validation loasses
    i = 0  # current iteration
    start = time.time()
    for epo in range(num_epo):

        cum_loss = 0  # Zero the cumulative sum
        for (x, y) in batch_generator(training_path, batch_size):
            out = model(x)  # calculate model outputs
            loss = loss_f(out, y)  # calculate loss

            cum_loss += loss.data.cpu().item()  # add current loss to cum_loss

            opt.zero_grad()  # zero old gradients
            loss.backward()  # calculate new gradients
            opt.step()  # perform optimization step

            i += 1
            if i % checkpoint_after == 0:
                val_loss = 0
                # measure loss on validation
                for (x, y) in test_batch_generator(test_path, batch_size):
                    out = model(x)
                    val_loss += loss_f(out, y).data.cpu().item()
                # add new data to lists
                losses_train.append(cum_loss / 1000)
                losses_val.append(val_loss / 1000)
                cum_loss = 0

                if np.argmin(losses_val) == len(losses_val) - 1:  # if new loss is the best
                    torch.save(model, weights_file)  # save model

                # if there was no improvement for drop_lr_after iterations
                if len(losses_val) - 1 - np.argmin(losses_val) > drop_lr_after:
                    model = torch.load(weights_file)  # load the best model
                    lr_start /= lr_deacy_rate  # reduce learning rate
                    opt = torch.optim.Adam(model.parameters(), lr=lr_start)

                # plot losses
                clear_output(True)
                print(f"Epoch {epo}: trining loss={losses_train[-1]}; test loss={losses_val[-1]}")
                print(f"Learninig rate: {lr_start}")
                end = time.time()
                print(f"Iteration took {end - start}s.")

                start = end
                plt.plot(losses_train, c="r")
                plt.plot(losses_val, c="g")
                plt.show()

                # If the model could not get better even after lr decay, then stop training
                if lr_start < 1e-6:
                    print(f"Early stop at {i}")
                    return


def plot_architecture(model):
    w = torch.zeros([1, 3, 32, 32])
    if is_cuda:
        w = w.cuda()
    return hl.build_graph(model, w)

def main():
    train_path = "cifar_data/train/"
    test_path = "cifar_data/test/"

    with open(test_path + 'test_batch', 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    labels = dict_[b"labels"]
    data = dict_[b"data"].reshape((-1, 3, 32, 32))

    model = models.LeNet(32)
    if is_cuda:
        model.to("cuda:0")

    train(model, train_path, test_path, 50, 16, "lenet.torch", lr_start=1e-3)


def plot_errors(model):
    pred = []
    for (x, _) in test_batch_generator(test_path, batch_size=20):
        pred += list(torch.argmax(model(x), 1).cpu().numpy())

    print(f"Top-1 Accuracy: {accuracy_score(pred, labels)}")
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix(pred, labels)*(1-np.eye(10)), annot=True)


if __name__ == '__main__':
    main()
