import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models
from torchvision import transforms
from torch.cuda import amp

from tqdm import tqdm
import time
import argparse

from sklearn.metrics import classification_report,roc_auc_score,average_precision_score, roc_curve,f1_score
from sklearn.model_selection import train_test_split
import dir_utils
from dir_utils import model_performance, getStat
from dataset import CustomDataset
from focal_loss import BinaryFocalLoss

from argparse import ArgumentParser
from lr_warmup_scheduler import GradualWarmupScheduler

from densenet import densenet121


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, default='echo/concat_pair_10.csv', help='Directory for data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--num_classes', type=int, default=1, help='Num of diagnostic classes')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')   # Out of memory...
    parser.add_argument("--test_batch_size",type=int, default=128, help='Test Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=33, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use_gpu', default=True, action='store_true', help='Use GPU')
    parser.add_argument('--model_path', type=str, default='echo/Dense_infer_Result', help='Path to saved model')
    parser.add_argument("--log_interval", type=str, default=40, help='log_interval')

    return parser.parse_args()


def main():
    ######### prepare environment ###########
    if torch.cuda.is_available() and args.use_gpu:

        device = torch.device('cuda')
        device_ids = [i for i in range(torch.cuda.device_count())]

        torch.cuda.set_device(5)   # NOTE!!

        print(device_ids)
        print('===> using GPU {} '.format(device_ids))
        print('current device: {}'.format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')
        print('===> using CPU !!!!!')

    epoch = args.epochs

    model_dir = args.model_path
    os.makedirs(model_dir, exist_ok = True)


    ######### dataset ###########
    data = pd.read_csv(args.csv_dir)
    _, _ = negBpow(data, '原始数据集', 'label')

    data_X = data.drop('label', axis = 1)
    data_y = data['label']

    # 划分训练集和测试集（比例为7:3）
    train_data, test_data, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=123, stratify=data_y)
    _, _ = negBpow(y_train, '训练集', '', True)
    _, _ = negBpow(y_test, '测试集', '', True)

    # 划分验证集（从之前划分的训练集中再划分）
    train_data, val_data, y_train, y_val = train_test_split(train_data, y_train, test_size=0.1428, random_state=123, stratify=y_train)
    _, _ = negBpow(y_train, '训练集', '', True)
    _, _ = negBpow(y_val, '验证集', '', True)   

    # 输出各个数据集的大小
    print("训练集大小：", len(train_data))
    print("验证集大小：", len(val_data))
    print("测试集大小：", len(test_data))

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((-0.037125014,),(5.2079306,))])   # self adaptive   
    
    # transform = transforms.Compose([transforms.ToTensor(),
    #                             transforms.Normalize((-0.054389194,),(3.5606365,))])  # self adaptive for II 

    train_dataset = CustomDataset(train_data, y_train, transform)
    # print(getStat(train_dataset))

    validation_dataset = CustomDataset(val_data, y_val, transform)
    test_dataset = CustomDataset(test_data, y_test, transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,
    #                          pin_memory=True)

    dataset_sizes = {'train': len(train_dataset),
                     'val': len(validation_dataset)}

    print('===> Loading datasets done')


    ######### model ###########
    model = densenet121()
    model = model.to(device)


    ######### optim ###########
    new_lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 15)

    # warmup_epochs = 1
    # scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch - warmup_epochs,
    #                                                         eta_min=1e-5)
    # scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
    #                                    after_scheduler=scheduler_cosine)
    # scheduler.step()  #调用scheduler.step(),则会改变optimizer中的学习率lr

    criterion = BinaryFocalLoss()
    # criterion = BinaryFocalLoss(alpha=0.3)


    print('===> model done')

    with open(os.path.join(model_dir, "log.csv"), "a") as f:
        epoch_resume = 0
        bestLoss = float("inf")
        try:   # 防止训练中断
            checkpoint = torch.load(os.path.join(model_dir, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint['epoch'] + 1
            bestLoss = checkpoint['best_loss']
            f.write("Resuming from epoch {}\n".format(epoch_resume))
            print("Epochs to resume: ", epoch_resume)
        except FileNotFoundError:
            f.write("Starting run from scratch\n")


        start_epoch = 1
        #torch.autograd.set_detect_anomaly(True)

        for epoch in range(start_epoch, epoch + 1):

            epoch_train_loss = 0

            train_pred_all, train_label_all = [], []
            val_pred_all, val_label_all = [], []
            val_pred_all_score = []

            #### train ####

            true_labels  = np.array([]) 
            pred_labels   = np.array([]) 
            prob_scores = np.array([])

            model.train()
            for batch_idx, (data, features, target, II, path) in enumerate(tqdm(train_loader)):
                data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
                II = II.to(device, dtype=torch.float)
                
                features = features.to(device, dtype=torch.float)

                optimizer.zero_grad()
                output = model(data)

                """pred(BCE loss)"""
                pred = torch.round(torch.sigmoid(output))

                """loss"""
                target = target.unsqueeze(-1)
                loss = criterion(torch.sigmoid(output), target)

                """update and save loss"""
                epoch_train_loss += loss.item()
                loss.backward()  # 更新权重
                optimizer.step()

                correct = 0
                total = target.size(0)
                correct += pred.eq(target.view_as(pred)).sum().item()
                accuracy = 100. * correct / total

                """auroc inputs"""
                train_pred_all.append(pred.cpu().detach().numpy())
                train_label_all.append(target.cpu().detach().numpy())


                if batch_idx % args.log_interval == 0:
                    # 1.
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss.item()))

                    # 2.
                    print('Train set:  batch loss: {:.4f}, Accuracy: {:.0f}% '.format(
                        loss.item(), accuracy))

                true_labels  = np.append(true_labels,np.array(target.cpu()))
                pred_labels  = np.append(pred_labels,np.array(pred.detach().cpu()))
                prob_scores  = np.append(prob_scores, np.array(torch.sigmoid(output).detach().cpu()))

            val_auroc = roc_auc_score(true_labels,prob_scores)
            FPR, TPR, thresholds = roc_curve(true_labels, prob_scores)

            """Youden Index"""
            J = TPR - FPR
            idx = np.argmax(J)
            best_thresh = thresholds[idx]

            print('Best Threshold=%f, sensitivity = %.3f, specificity = %.3f, J=%.3f' % (best_thresh, TPR[idx], 1-FPR[idx], J[idx]))


            train_loss_mean = epoch_train_loss / dataset_sizes['train']

            train_pred_all = np.concatenate(train_pred_all, axis=0)
            train_label_all = np.concatenate(train_label_all, axis=0)

            #### Evaluation ####

            true_labels  = np.array([])
            pred_labels   = np.array([]) 
            prob_scores = np.array([])       

            with torch.no_grad():
                model.eval()
                epoch_val_loss = 0
                val_sen,val_spe ,val_ppv, val_npv = [], [], [], []

                for batch_idx, (data, features, target, II, path) in enumerate(tqdm(validation_loader)):
                    data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
                    II = II.to(device, dtype=torch.float)

                    features = features.to(device, dtype=torch.float)

                    output = model(data)
                    outputs_score = output.clone()

                    """pred(BCE loss)"""
                    pred = (torch.sigmoid(output) > best_thresh).int()

                    """loss"""
                    loss = criterion(torch.sigmoid(output), target.unsqueeze(-1))
                    epoch_val_loss += loss.item()

                    """auroc inputs"""
                    val_pred_all.append(pred.squeeze(-1).cpu().detach().numpy())
                    val_pred_all_score.append(outputs_score.cpu().detach().numpy())
                    val_label_all.append(target.cpu().detach().numpy())


                    true_labels  = np.append(true_labels,np.array(target.cpu()))
                    pred_labels  = np.append(pred_labels,np.array(pred.squeeze(-1).cpu()))
                    prob_scores  = np.append(prob_scores, np.array(torch.sigmoid(output).cpu()))

                val_auroc = roc_auc_score(true_labels,prob_scores)
                val_auprc = average_precision_score(true_labels,prob_scores, average=None)
                FPR, TPR, thresholds = roc_curve(true_labels, prob_scores)
                

                """calculate sen & spe"""
                _, _, _, _, sen, spe, ppv, npv = model_performance(true_labels,pred_labels)
                val_sen.append(sen)
                val_spe.append(spe)
                val_ppv.append(ppv)
                val_npv.append(npv)
                
                """calcuate precision, recall, f1"""
                f1 = f1_score(true_labels, pred_labels)

                f.write("{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                                val_auroc,
                                                                val_auprc,
                                                                f1,
                                                                sen,
                                                                spe,
                                                                ppv,
                                                                npv,
                                                                best_thresh))
                f.flush()


                val_pred_all = np.concatenate(val_pred_all, axis=0)
                val_pred_all_score = np.concatenate(val_pred_all_score, axis=0)
                val_label_all = np.concatenate(val_label_all, axis=0)


            val_loss_mean = epoch_val_loss / dataset_sizes['val']
            scheduler.step()


            print("------------------------------------------------------------------")
            print("val loss: {:.6f} \t LearningRate {:.8f}".format(val_loss_mean, scheduler.get_lr()[0]))
            print('Valid set: AUROC: {:.4f}'.format(val_auroc))
            print('Valid set: AUPRC: {:.4f}'.format(val_auprc))
            print('Valid set: F1 score: {:.4f}'.format(f1))
            print('Valid set: Sensitivity = Recall: {:.4f}'.format(sen))
            print('Valid set: Specificity: {:.4f}'.format(spe))
            print('Valid set: Precision = PPV: {:.4f}'.format(ppv))
            print('Valid set: NPV: {:.4f}'.format(npv))
            print('-------------------------------------------')
            

            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_loss': bestLoss,
                'loss': val_loss_mean,
                'opt_dict': optimizer.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
                'Youden Index': best_thresh
            } 

            torch.save(save, os.path.join(model_dir, "checkpoint.pt"))
            if val_loss_mean < bestLoss:   # based on the loss value   
                torch.save(save, os.path.join(model_dir, "best.pt"))
                bestLoss = val_loss_mean



if __name__ == "__main__":

    args = parse_args()

    main()