import torch
import os, random
import numpy as np
import pandas as pd
from scipy import stats
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

seed_num = 42
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)


def seed_worker(worker_id):
    numpy.random.seed(seed_num)
    random.seed(seed_num)

g = torch.Generator()
g.manual_seed(seed_num)


class SleepApnea_dataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        
    def __getitem__(self, index):
        return self.data[index], self.label[index]
        
    def __len__(self):
        return len(self.data)
        
        

class Dataset:
    def __init__(self, dataset, _batch_size, X_train_norm, y_train, X_val_norm, y_val, X_raw_norm, y_raw):
        super(Dataset, self).__init__()
                
        if dataset == "sleepApnea":
            X_train_norm = np.concatenate([X_train_norm, np.zeros((X_train_norm.shape[0], 6))], axis=1).reshape(X_train_norm.shape[0], 1, 18, 18)
            X_val_norm = np.concatenate([X_val_norm, np.zeros((X_val_norm.shape[0], 6))], axis=1).reshape(X_val_norm.shape[0], 1, 18, 18)
            
            X_train_norm = X_train_norm.astype(np.float32)
            X_val_norm = X_val_norm.astype(np.float32)

            print("Training data :", X_train_norm.shape, "Validation data :", X_val_norm.shape) #, "Raw data :", X_raw_norm.shape)
            print()
            
            self.train_loader = DataLoader(dataset = SleepApnea_dataset(X_train_norm, y_train), batch_size=_batch_size, shuffle=True,
            worker_init_fn=seed_worker, generator=g)
            self.val_loader = DataLoader(dataset = SleepApnea_dataset(X_val_norm, y_val), batch_size=_batch_size, shuffle=False)
            

    def getNormalizedData(self, Type, X_train, X_val):
      """
      args:
      Type: Minmax or Standard scaling technique
      X_train: Training data
      X_val: Validation data
    
      return X_train_norm, X_val_norm, X_test_norm # Normalized
      """
    
      if Type == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X_train)
      elif Type == "std":
        scaler = StandardScaler()
        scaler.fit(X_train)
      else:
        scaler = RobustScaler()
        scaler.fit(X_train)
    
      X_train_norm, X_val_norm = scaler.transform(X_train), scaler.transform(X_val)
    
      return X_train_norm, X_val_norm, scaler
    
    

      
      
    

      
    
    
    

        


