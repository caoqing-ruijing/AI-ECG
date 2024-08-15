import sklearn
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import torch
from tqdm import tqdm
from natsort import natsorted
from glob import glob
import shutil


def model_performance(labels, preds):
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    sen = (tp / (tp + fn)) * 100
    spe = (tn / (fp + tn)) * 100

    ppv = (tp / (tp + fp)) * 100
    npv = (tn / (fn + tn)) * 100

    return tn, fp, fn, tp, sen, spe, ppv, npv



def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            os.mkdir(path)
    else:
        os.mkdir(paths)

def mkdir_without_del(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdir_with_del(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def get_last_path(path, session):
	x = natsorted(glob(os.path.join(path,'*%s'%session)))[-1]
	return x

def getStat(train_data):
    print("compute mean and std for training data")
    print("train data len:", len(train_data))

    train_loader = torch.utils.data.DataLoader(train_data, shuffle=False, num_workers=0, pin_memory=True)
    means = torch.zeros(1)
    stds = torch.zeros(1)

    for (data, features, target, II, path) in tqdm(train_loader):
        
        means[0] += data[:, 0, :, :].mean()
        stds[0] += data[:, 0, :, :].std()

    means.div_(len(train_data))
    stds.div_(len(train_data))

    return list(means.numpy()), list(stds.numpy())