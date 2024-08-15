import pandas as pd
import numpy as np
import os
import random

import torch
from torch import nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as transforms

import sklearn
from sklearn.model_selection import train_test_split

from dataset import CustomDataset
from densenet import densenet121

from criterion import WeightedMSELoss, HuberLoss
from plt import plt_contour, plot_contour, plot_sns_contour, plot_scatter_alpha, plot_scatter_kde, plot_hexbin, plot_scatter_cmap, plot_altman, plot_scatter_test
from dir_utils import getStat


def negBpow(data, name, label, seqflag=False):
    dc = len(data)
    if seqflag:
        poscount = len(data[data == 1])
        negcount = len(data[data == 0])
    else:
        poscount = len(data[data[label] == 1])
        negcount = len(data[data[label] == 0])
    pos_rate, neg_rate = poscount / dc, negcount / dc
    print(' {}: 正样本数：{}，负样本数：{}，正负样本比: {} : {}'.format(name, poscount, negcount, 1, np.around(negcount / poscount, decimals=4)))
    # print('   正样本占比：{}，负样本占比：{}'.format(pos_rate, neg_rate))
    return poscount, negcount


data = pd.read_csv("echo/concat_pair_11.csv", low_memory = False)

# devide and distribution
_, _ = negBpow(data, '原始数据集', 'label')

data_X = data.drop('label', axis = 1)
data_y = data['label']

# 划分训练集和测试集（默认比例为7:3）
train_data, test_data, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=123, stratify=data_y)
_, _ = negBpow(y_train, '训练集', '', True)
_, _ = negBpow(y_test, '测试集', '', True)

# 划分验证集（从之前划分的训练集中再划分）
train_data, val_data, y_train, y_val = train_test_split(train_data, y_train, test_size=0.1428, random_state=123, stratify=y_train)
_, _ = negBpow(y_train, '训练集', '', True)
_, _ = negBpow(y_val, '验证集', '', True)  

# 从每个数据集中提取 LVEF ≤ 50% 的样本
train_data_subgroup_low = train_data[y_train == 1]
val_data_subgroup_low = val_data[y_val == 1]
test_data_subgroup_low = test_data[y_test == 1]

# 从每个数据集中随机提取与LVEF ≤ 50%等量的LVEF > 50%的样本
train_data_subgroup_high = train_data[y_train == 0].sample(n=int(len(train_data_subgroup_low)), random_state=123)
val_data_subgroup_high = val_data[y_val == 0].sample(n=int(len(val_data_subgroup_low)), random_state=123)
test_data_subgroup_high = test_data[y_test == 0].sample(n=int(len(test_data_subgroup_low)), random_state=123)

# 合并数据集，使得每个集合中LVEF ≤ 50%和LVEF > 50%的数据样本比例保持1:1
train_data_balanced = pd.concat([train_data_subgroup_low, train_data_subgroup_high])
val_data_balanced = pd.concat([val_data_subgroup_low, val_data_subgroup_high])
test_data_balanced = pd.concat([test_data_subgroup_low, test_data_subgroup_high])

# 输出各数据集的大小
print("平衡后的训练集大小:", len(train_data_balanced))
print("平衡后的验证集大小:", len(val_data_balanced))
print("平衡后的测试集大小:", len(test_data_balanced))


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((-0.037125014,),(5.2079306,))])   # 一个通道这样写，mean和std

# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize((-0.056172576,),(7.936976,))])   # self adaptive


train_dataset = CustomDataset(train_data_balanced, y_train, transform)
validation_dataset = CustomDataset(val_data_balanced, y_val, transform)
test_dataset = CustomDataset(test_data_balanced, y_test, transform)   # fixed

# print(getStat(train_dataset))


"""hyper parameters"""
# learning rate
lr = 1e-3  
weight_decay = 1e-4

batch_size = 32   # global
val_batch_size = 128
test_batch_size = 128


epochs = 100
no_cuda = False


"""GPU"""
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

torch.cuda.set_device(3)

kwargs = {'num_workers':0,'pin_memory':True} if use_cuda else {}

print('--------------------------------------')
print('torch :',torch.__version__)
print('device:',device)
print('cuda index:', torch.cuda.current_device())
print('gpu :', torch.cuda.device_count())
print('graphic name:', torch.cuda.get_device_name())



"""Train"""
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size = batch_size,
    shuffle = True, 
    **kwargs
)

"""Validation"""
validation_loader = torch.utils.data.DataLoader(
    dataset=validation_dataset,
    batch_size = val_batch_size,
    shuffle = True,
    **kwargs
)

"""Test"""
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size = test_batch_size,
    shuffle = True,
    **kwargs
)


def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(s)
    random.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)


from tqdm import tqdm

def run_epoch(model, dataloader, train, optimizer, device):
    model.train(train)
    
    total_loss = 0
    y, yhat = [], []
    # criterion = WeightedMSELoss().to(device)
    # criterion = HuberLoss(delta=1.0).to(device)
    
    with torch.set_grad_enabled(train):
        with tqdm(total = len(dataloader)) as progressbar:
            for (data, features, target, leadII) in dataloader:
                
                y.append(target.numpy())
                data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
                features = features.to(device, dtype=torch.float)
                
                output = model(data)   
                
                yhat.append(output.view(-1).to("cpu").detach().numpy())  
                loss = nn.functional.mse_loss(output.view(-1), target)
                # loss = criterion(output.view(-1), target)   # treat imbalanced data
                
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                total_loss += loss.item()   
                
                progressbar.set_postfix_str("{:.2f} ({:.2f})".format(total_loss, loss.item()))
                progressbar.update()
                
    yhat = np.concatenate(yhat)
    y = np.concatenate(y)
    
    return total_loss / len(dataloader), yhat, y


import time

def run_train(result, device, model, optimizer, lr_scheduler, bestLoss, epoch_resume, f):
    for epoch in range(epoch_resume, epochs):
        print("Epoch #{}".format(epoch), flush = True)
        for phase in ['train', 'val']:
            start_time = time.time()
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
            
            dataloader = train_loader
            if phase == 'val':
                dataloader = validation_loader
            loss, yhat, y = run_epoch(model, dataloader, phase == 'train', optimizer, device)
            f.write("{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                             phase,
                                                             loss,
                                                             sklearn.metrics.r2_score(y, yhat),
                                                             time.time() - start_time,
                                                             y.size,
                                                             sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                             sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                             batch_size))

            
            f.flush()

        lr_scheduler.step()
        
        save = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_loss': bestLoss,
            'loss': loss,
            'r2': sklearn.metrics.r2_score(y, yhat),
            'opt_dict': optimizer.state_dict(),
            'scheduler_dict': lr_scheduler.state_dict()
        }
        torch.save(save, os.path.join(result, "checkpoint.pt"))
        if loss < bestLoss:
            torch.save(save, os.path.join(result, "best.pt"))
            bestLoss = loss


import math
def bootstrap_metric(arg1, arg2, fun, num_samples=10000):
    results = []
    arg1, arg2 = np.array(arg1), np.array(arg2)

    for _ in range(num_samples):
        index = np.random.choice(len(arg1), len(arg1))
        results.append(fun(arg1[index], arg2[index]))

    results = sorted(results)
    percentile_05 = results[round(0.05 * len(results))]
    percentile_95 = results[round(0.95 * len(results))]

    return fun(arg1, arg2), percentile_05, percentile_95
    
def run_test(output, device, model, f):
    if epochs != 0:
        checkpoint = torch.load(os.path.join(output, "best.pt"))
        model.load_state_dict(checkpoint['state_dict'])
        print(os.path.join(output, "best.pt"))
        f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))
        f.flush()

    split = "test"
    set_seed(0)
    dataloader = test_loader
    loss, yhat, y = run_epoch(model, dataloader, False, None, device)   # no optimizer


    r2  = bootstrap_metric(y, yhat, sklearn.metrics.r2_score)
    mae = bootstrap_metric(y, yhat, sklearn.metrics.mean_absolute_error)
    rmse = tuple(map(math.sqrt, bootstrap_metric(y, yhat, sklearn.metrics.mean_squared_error)))

    # plot_scatter_cmap(y, yhat)
    # plot_scatter_test(y, yhat)

    print("R2: ", sklearn.metrics.r2_score(y, yhat))

    f.write("{} R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *r2))
    f.write("{} MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *mae))
    f.write("{} RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *rmse))
    f.flush()


"""main"""
import argparse
import torch.distributed as dist

my_parser = argparse.ArgumentParser(description = 'Run script')

my_parser.add_argument('--exp_no',
                      type = str,
                      default="123",
                      help = 'Experiment number for wandb')

my_parser.add_argument('--exp_name',
                      type = str,
                      default="ECG2EF",
                      help = 'Experiment name for wandb')

my_parser.add_argument('--weights',
                      type = str,
                      default="None",
                      help = 'Path to checkpoint to load')

my_parser.add_argument('--epochs',
                      type = int,
                      default="150",
                      help = 'Number of epochs to train')

my_parser.add_argument('--optimizer_name',
                      type = str,
                      default="Adam",
                      help = 'Optimizer name')

my_parser.add_argument('--batch_size',
                      type = int,
                      default="32",
                      help = 'Batch size')


args = my_parser.parse_args()
print("Exp Name: ", args.exp_name)
print("Exp No.: ", args.exp_no)
print("Epochs: ", epochs)


set_seed(0)


# result = 'echo/regression/Dense_WMSE_Result'
# result = "echo/regression/Dense_Log_Result"
# result = "echo/regression/Dense_MAE_Result"
# result = "echo/regression/Dense_Sub1_Result"
# result = "echo/regression/Dense_Sub3_Result"
result = "echo/regression/Dense_Sub9_Result"
# result = "echo/regression/Dense_Sub10_Result"
# result = "echo/regression/Dense_Sub12_Result"
os.makedirs(result, exist_ok = True)


model = densenet121()
model.to(device)


optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
# 学习率调度器，即在训练过程中动态地调整学习率
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 15)   # 15表示每隔15个epoch将学习率衰减一次


with open(os.path.join(result, "log.csv"), "a") as f:
    epoch_resume = 0
    bestLoss = float("inf")
    try:   # 防止训练中断
        checkpoint = torch.load(os.path.join(result, "checkpoint.pt"))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_dict'])
        epoch_resume = checkpoint['epoch'] + 1
        bestLoss = checkpoint['best_loss']
        f.write("Resuming from epoch {}\n".format(epoch_resume))
        print("Epochs to resume: ", epoch_resume)
    except FileNotFoundError:
        f.write("Starting run from scratch\n")
    
    if epoch_resume < epochs:
        run_train(result, device, model, optimizer, lr_scheduler, bestLoss, epoch_resume, f)
        
    # print(model)

    run_test(result, device, model, f)