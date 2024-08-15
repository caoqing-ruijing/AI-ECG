import os
import json
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class CustomDataset(Dataset):

    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    def __getitem__(self, idx):
        
        """get data"""
        tempt = self.data.iloc[idx]
        Path = tempt['ECG_ID']
        with open(os.path.join('ecg_data_json', Path+'.json'), 'r') as json_file:
            ecg = json.load(json_file)
        
        LeadI   = ecg['I']['value']
        LeadII  = ecg['I I']['value']
        LeadIII = ecg['III']['value']
        V1      = ecg['V1']['value']
        V2      = ecg['V2']['value']
        V3      = ecg['V3']['value']
        V4      = ecg['V4']['value']
        V5      = ecg['V5']['value']
        V6      = ecg['V6']['value']
        aVL     = ecg['aVL']['value']
        aVR     = ecg['aVR']['value']
        aVF     = ecg['aVF']['value']
        
        II  = ecg['II']['value'][500:-500]   # 4000 samples
        new_II = [float(x) for x in II]   # NOTE!!

        # data = np.vstack([LeadI, LeadII, LeadIII,aVR, aVL, aVF, V1, V2, V3, V4, V5, V6])
        ecg = np.vstack([V1, V2, V3, V4, V5, V6, aVL, LeadI, aVR, LeadII, aVF, LeadIII])
        
        ecg = np.expand_dims(ecg, axis=-1)   # Note!!

        up_II = np.vstack([new_II])
        data_II = np.expand_dims(up_II, axis=-1)


        # Normalization
        if self.transform is not None:
            ecg = self.transform(ecg)
            data_II = self.transform(data_II)
        

        sex = float(tempt['Gender'])
        age = float(tempt['Age'])
        
        """ecg features"""
        PRInterval      = float(tempt['PR']) 
        QRSDuration     = float(tempt['QRS'])
        QTInterval      = float(tempt['QT'])
        QTCorrected     = float(tempt['QTc'])

        
        # features = np.array([sex,age,PRInterval,QRSDuration,
        #                      QTInterval,QTCorrected])
        features = np.array([sex,age])   # only for sex and age!!
        features = np.expand_dims(features, axis=-1)  
        

        scaler2 = MinMaxScaler()
        features = scaler2.fit_transform(features)
        features = features.reshape(-1)


        """get label"""
        label = float(self.label.iloc[idx])  


        return ecg, features, label, data_II, Path
    
    
    def __len__(self):
        return len(self.data)