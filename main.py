import joblib
import argparse, os
import numpy as np
import pandas as pd
import torch, random
import torch.nn as nn
from scipy import stats
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from OxyCaps import OxyNet
from data_loader import Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import confusion_matrix, classification_report

class Config:
    def __init__(self, dataset=''):
        
        if dataset == 'sleepApnea':
            
            # Parameters for OxyCaps network
            
            # CNN (cnn)
            self.cnn_in_channels = 1
            self.cnn_out_channels = 32
            self.cnn_kernel_size = 3

            # Primary Capsule (pc)
            self.pc_num_capsules = 4
            self.pc_in_channels = 32
            self.pc_out_channels = 32
            self.pc_kernel_size = 3
            self.pc_num_routes = 32 * 6 * 6

            # High-Level Capsule (hc)
            self.hc_num_capsules = 10
            self.hc_num_routes = 32 * 6 * 6
            self.hc_in_channels = 4
            self.hc_out_channels = 16

            # Dimension of input features
            self.input_width = 18
            self.input_height = 18



def train(model, optimizer, train_loader, epoch):
    
    """
    args:
    model: OxyNet model
    optimizer: Adam or AdamW
    train_loader: Data loader for training set
    epoch: Epoch number
    """
    oxy_net = model
    oxy_net.train()
    n_batch = len(list(enumerate(train_loader)))
    total_loss = 0

    for batch_id, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = oxy_net(data)
        loss = oxy_net.loss(output, target)
        loss.backward()
        optimizer.step()
        correct = sum(np.argmax(output.data.cpu().numpy(), 1) == target.data.cpu().numpy())
        train_loss = loss.item()
        total_loss += train_loss
        
        if (batch_id) % 100 == 0:
            tqdm.write("Epoch: [{}/{}], Batch: [{}/{}], train accuracy: {:.6f}, train loss: {:.6f}".format(
                epoch,
                N_EPOCHS,
                batch_id + 1,
                n_batch,
                correct / float(BATCH_SIZE),
                train_loss / float(len(data))
                ))
                
    tqdm.write('Epoch: [{}/{}], train loss: {:.6f}'.format(epoch,N_EPOCHS,total_loss / len(train_loader.dataset)))
    return total_loss / len(train_loader.dataset)


def test(oxy_net, test_loader, epoch):
    """
    args:
    oxy_net: OxyNet model
    test_loader: Data Loader for test set
    epoch: Epoch number
    """
    oxy_net.eval()
    test_loss = 0
    
    correct = 0
    for batch_id, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = oxy_net(data)
        loss = oxy_net.loss(output, target)
        test_loss += loss.item()
        correct += sum(np.argmax(output.data.cpu().numpy(), 1) == target.data.cpu().numpy())
    
    tqdm.write(
    "Epoch: [{}/{}], test accuracy: {:.6f}, loss: {:.6f}".format(epoch, N_EPOCHS, correct / len(test_loader.dataset),
                                                              test_loss / len(test_loader)))
    return test_loss / len(test_loader)


  
  
def getPredAndLabelValues(datasetLoader, oxy_net):
    """
    args:
    datasetLoader: Data loader object for either training or validation or test set
    oxy_net: OxyNet model
    
    return prediction,label
    """
    pred = []
    label = []

    for batch_id, (data, target) in enumerate(datasetLoader):
        data, target = data.to(device), target.to(device)
            
        pred += np.argmax(oxy_net(data).data.cpu().numpy(), 1).tolist()
        label += target.data.cpu().numpy().tolist()
        
    return pred, label



def getNormalizedData(Type, x_train, x_val):
    """
    args:
    Type: Minmax or Standard or Robust scaling technique
    X_train: Training data
    X_val: Validation data
    
    return X_train_norm, X_val_norm, Scaler object # Normalized data
    """
    
    if Type == "minmax":
        scaler = MinMaxScaler()
        scaler.fit(x_train)
    elif Type == "std":
        scaler = StandardScaler()
        scaler.fit(x_train)
    else:
        scaler = RobustScaler()
        scaler.fit(x_train)
    
    x_train_norm, x_val_norm = scaler.transform(x_train), scaler.transform(x_val)
    
    return x_train_norm, x_val_norm, scaler

  
def mappingLabel(x):
    """
    99-96 - Normal --> 2
    96-92 - Low    --> 1
    88-92 - Danger --> 0
    """
    
    if 88 <= x <= 91:
        return 0
    elif 92 <= x <= 95:
        return 1
    elif 96 <= x <= 99:
        return 2
        
        

def getAugmentedFeatures(df, cols_to_work):
    
    patient_records = []
    for patient in df['patient_ID'].unique():
        patient_df = df[df['patient_ID'] == patient]
        patient_df = patient_df.sort_values('time_stamp').reset_index(drop=True)
    
        # Features generation
        for col in cols_to_work:
            if col in ['amp_exhale_lag_125', 'amp_exhale_lag_250', 'amp_inhale_lag_125', 'amp_inhale_lag_250', 'dur_exhale_lag_125', 'dur_exhale_lag_250', 'dur_inhale_lag_125', 'dur_inhale_lag_250', 'rr_lag_125', 'rr_lag_250']:
                continue
            # Taking previous timestamp as a feature
            patient_df[col+'_shift_1' ] = patient_df[col].shift(periods=1)
            patient_df[col+'_shift_2' ] = patient_df[col].shift(periods=2)
            patient_df[col+'_shift_3' ] = patient_df[col].shift(periods=3)
            
            patient_df = patient_df.fillna(0)
            
            # Computing difference between feature at t and (t-1)
            patient_df[col+'_diff_1' ] = patient_df[col] - patient_df[col+'_shift_1' ]
            patient_df[col+'_diff_2' ] = patient_df[col] - patient_df[col+'_shift_2' ]
            patient_df[col+'_diff_3' ] = patient_df[col] - patient_df[col+'_shift_3' ]
            
            # Computing difference w.r.t. mean, median, min, max
            patient_df[col+'_min' ] = patient_df[col] - patient_df[col].min()
            patient_df[col+'_max' ] = patient_df[col].max() - patient_df[col]
            patient_df[col+'_mean' ] = patient_df[col] - patient_df[col].mean()
            patient_df[col+'_median' ] = patient_df[col] - patient_df[col].median()
        
        patient_records.append(patient_df)

    patient_records = pd.concat(patient_records).reset_index(drop=True)
    
    return patient_records




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description= "Hypoxemia classification task")
    parser.add_argument('--epoch', type = int, help = "Number of epochs.", default = 2)
    parser.add_argument('--batchSize', type = int, help = "Batch Size.", default = 512)
    parser.add_argument('--learningRate', type = float, help = "Learning Rate.", default = 0.0005) # 0.0001
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Fetching values from user's input
    N_EPOCHS = args.epoch
    BATCH_SIZE = args.batchSize
    LEARNING_RATE = args.learningRate

    # For reproducibility 
    seed_num = 42
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic =True
    
    USE_CUDA = True if torch.cuda.is_available() else False
    
    print("Loading Sleep Apnea Data ...", "\n")
    columns_to_remove = ["time_stamp_corrected", "time_stamp", 'location', "calibration", 'category', 'patient_ID', 'SpO2(%)']
    
    # Balanced data
    df = pd.read_hdf("PatientDataClass_25Hz_88_99_feats_44_lag.h5")
    cols_to_work = list(set(df.columns) - set(columns_to_remove + ["SpO2(%)", 'patient_ID']))
    
    df = getAugmentedFeatures(df, cols_to_work)
    df['category'] = df['SpO2(%)'].apply(lambda row:mappingLabel(row))
    
    # Balanced data and their label
    label = df['category'].values
    df = df.drop(columns_to_remove, axis=1)


    for i in range(5):
        print(f"Fold {i+1}:")
        dataset = "sleepApnea"
        config = Config(dataset)
        
        X_train, X_val, y_train, y_val = train_test_split(df, label, test_size=0.10, stratify = label, random_state=i)
        X_train_norm, X_val_norm, scaler = getNormalizedData("std", X_train, X_val)
            
        joblib.dump(scaler, os.getcwd() +"/Scaler_fold_v_" + str(i+1) + ".pkl") 
        
        sleepApnea = Dataset(dataset, BATCH_SIZE, X_train_norm, y_train, X_val_norm, y_val, None, None)
        
        oxy_net = OxyNet(config)
        oxy_net = torch.nn.DataParallel(oxy_net)
        oxy_net = oxy_net.to(device)
        oxy_net = oxy_net.module
    
        optimizer = torch.optim.AdamW(oxy_net.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)
        
        for e in range(1, N_EPOCHS + 1):
            avg_train_loss = train(oxy_net, optimizer, sleepApnea.train_loader, e)
            avg_val_loss = test(oxy_net, sleepApnea.val_loader, e)
            scheduler.step(avg_val_loss)
            
        # Saving the model
        torch.save(oxy_net.state_dict(), os.getcwd()+ "/Model_fold_v_" + str(i+1)+ ".pt")
            
        pred_val, label_val = getPredAndLabelValues(sleepApnea.val_loader, oxy_net)
        pred_train, label_train = getPredAndLabelValues(sleepApnea.train_loader, oxy_net)
        
        # Confusion Matrix
        print("Confusion Matrix for Training data:")
        print(confusion_matrix(label_train, pred_train))
        print()
        print("Classification Report for Training data:")
        print(classification_report(label_train, pred_train))
        print()
        print()
        print("*"*51)
        print()
        print()
        print("Confusion Matrix for Validation data:")
        print(confusion_matrix(label_val, pred_val))
        print()
        print("Classification Report for Validation data:")
        print(classification_report(label_val, pred_val))
        print()
