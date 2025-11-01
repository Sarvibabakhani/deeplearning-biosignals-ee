import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from itertools import combinations
from collections import defaultdict


from ResNet_modules import ResNet1D, root_mean_squared_error
from Train_evaluate_ResNet import train_model, evaluate_model,evaluate_model_by_activity

#---------------------- datasets----------------------------------------------------
#Here you should place the path of the dataset belonging to each subject. 

# sub1 = pd.read_csv('PATH to dataset1')
# sub2 = pd.read_csv('PATH to dataset2')
# sub3 = pd.read_csv('PATH to dataset3')
# sub4 = pd.read_csv('PATH to dataset4')
# sub5 = pd.read_csv('PATH to dataset5')
# sub6 = pd.read_csv('PATH to dataset6')
# sub7 = pd.read_csv('PATH to dataset7')
# sub8 = pd.read_csv('PATH to dataset8')
# sub9 = pd.read_csv('PATH to dataset9')
# sub10 = pd.read_csv('PATH to dataset10')
#---------------------------------------------------------------------------------------------------------S


def run_resnet(sig,epochs=100):
    
    # print(len(sig))
    rmse_list = []
    #Cross Validation
    df_list2 = [sub1, sub2, sub3, sub4, sub5, sub6, sub7, sub8, sub9, sub10]
    for i, test_df in enumerate(df_list2):
                
            
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8' #deterministic CuBLAS behavior
            
            # Set random seeds for reproducibility
            np.random.seed(42)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True)
            
            
            num_time_steps = 10  
            
            test_df = df_list2[i]
            train_val_df = pd.concat(df_list2[:i] + df_list2[i + 1:], ignore_index=True)
            X_train_val = train_val_df[sig].values
            y_train_val = train_val_df[['Interpolated Values_Enregy expendiures']].values
            X_test = test_df[sig].values
            y_test = test_df[['Interpolated Values_Enregy expendiures']].values
            
            # (samples, 1, time steps)
            num_samples = len(X_train_val) // num_time_steps * num_time_steps
            X_train_val = X_train_val[:num_samples].reshape(-1, len(sig), num_time_steps)
            y_train_val = y_train_val[:num_samples].reshape(-1, num_time_steps)

            np.random.seed(42)
            indices = np.arange(X_train_val.shape[0])
            np.random.shuffle(indices)
            X_train_val = X_train_val[indices]
            y_train_val = y_train_val[indices]
            
            # Split 
            val_size = int(0.15 * len(X_train_val))
            X_val, X_train = X_train_val[:val_size], X_train_val[val_size:]
            y_val, y_train = y_train_val[:val_size], y_train_val[val_size:]
            
            # Normalize data
            num_test_samples = len(X_test) // num_time_steps * num_time_steps
            X_test = X_test[:num_test_samples].reshape(-1, len(sig), num_time_steps)
            y_test = y_test[:num_test_samples].reshape(-1, num_time_steps)
        
            scaler = StandardScaler()
            # if len(sig)!= 1:
            X_train_reshaped = X_train.transpose(0, 2, 1).reshape(-1, len(sig))
            X_val_reshaped = X_val.transpose(0, 2, 1).reshape(-1, len(sig))

            X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(-1, num_time_steps, len(sig)).transpose(0, 2, 1)
            X_val_scaled = scaler.transform(X_val_reshaped).reshape(-1, num_time_steps, len(sig)).transpose(0, 2, 1)

            X_test_reshaped = X_test.transpose(0, 2, 1).reshape(-1, len(sig))
            X_test_scaled = scaler.transform(X_test_reshaped).reshape(-1, num_time_steps, len(sig)).transpose(0, 2, 1)
            # else:    
            #     X_train_scaled = scaler.fit_transform(X_train.reshape(-1, num_time_steps)).reshape(-1, 1 , num_time_steps)
            #     X_val_scaled = scaler.transform(X_val.reshape(-1, num_time_steps)).reshape(-1, 1, num_time_steps)

            #     X_test_scaled = scaler.transform(X_test.reshape(-1, num_time_steps)).reshape(-1, 1, num_time_steps)
            
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
            
            batch_size = 32
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = ResNet1D(input_channels=len(sig), output_size=num_time_steps).to(device)
            
            # Training 
            criterion = root_mean_squared_error
            optimizer = optim.Adam(model.parameters(), lr=0.001 ,betas=(0.89 ,0.999))
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=10)
            early_stopping_patience = 20
            
    
            train_model(model, train_loader, val_loader, device,optimizer,criterion,scheduler,early_stopping_patience = early_stopping_patience, epochs=epochs)
                
            rmse = evaluate_model(model, test_loader,device,criterion)
            rmse_list.append(rmse)
            
        
    average_rmse = np.mean(rmse_list)
    return rmse_list, sig, average_rmse




def run_resnet_perActivity(sig,epochs=100):
        
        activity_rmse_across_folds = defaultdict(list)
        rmse_list = []
        
        #  Cross Validation 
        df_list2 = [sub1, sub2, sub3, sub4, sub5, sub6, sub7, sub8, sub9, sub10]
        for i, test_df in enumerate(df_list2):
                    
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
                
                
                np.random.seed(42)
                torch.manual_seed(42)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(42)
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    torch.use_deterministic_algorithms(True)
                
                
                num_time_steps = 10  
                
                test_df = df_list2[i]
                train_val_df = pd.concat(df_list2[:i] + df_list2[i + 1:], ignore_index=True)
                
                X_train_val = train_val_df[sig].values
                y_train_val = train_val_df[['Interpolated Values_Enregy expendiures']].values

                num_samples = len(X_train_val) // num_time_steps * num_time_steps
                X_train_val = X_train_val[:num_samples].reshape(-1, len(sig), num_time_steps)
                y_train_val = y_train_val[:num_samples].reshape(-1, num_time_steps)
                
                indices = np.arange(X_train_val.shape[0])
                np.random.shuffle(indices)
                X_train_val = X_train_val[indices]
                y_train_val = y_train_val[indices]
                
                val_size = int(0.15 * len(X_train_val))
                X_val, X_train = X_train_val[:val_size], X_train_val[val_size:]
                y_val, y_train = y_train_val[:val_size], y_train_val[val_size:]

                scaler = StandardScaler()
                # if len(sig)!= 1:
                X_train_reshaped = X_train.transpose(0, 2, 1).reshape(-1, len(sig))
                X_val_reshaped = X_val.transpose(0, 2, 1).reshape(-1, len(sig))

                X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(-1, num_time_steps, len(sig)).transpose(0, 2, 1)
                X_val_scaled = scaler.transform(X_val_reshaped).reshape(-1, num_time_steps, len(sig)).transpose(0, 2, 1)

               
                # else:    
                #     X_train_scaled = scaler.fit_transform(X_train.reshape(-1, num_time_steps)).reshape(-1, 1 , num_time_steps)
                #     X_val_scaled = scaler.transform(X_val.reshape(-1, num_time_steps)).reshape(-1, 1, num_time_steps)
    

                
                X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
                X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

                
                batch_size = 32
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = ResNet1D(input_channels=len(sig), output_size=num_time_steps).to(device)
                
                # Training 
                criterion = root_mean_squared_error
                optimizer = optim.Adam(model.parameters(), lr=0.001 ,betas=(0.89, 0.999))
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=10)
                early_stopping_patience = 20
                
                
                train_model(model, train_loader, val_loader, device,optimizer,criterion,scheduler,early_stopping_patience = early_stopping_patience, epochs=epochs)
              
                activity_rmse_across_folds= evaluate_model_by_activity(model, test_df,device,criterion,sig=sig, batch_size=batch_size , num_time_steps=num_time_steps, scaler=scaler ,activity_column='Activity Code',activity_rmse_across_folds =activity_rmse_across_folds) 
                
        average_rmse_per_activity = {
            activity: np.mean(rmses) for activity, rmses in activity_rmse_across_folds.items()
        }
        print(sig)
        print("Average RMSE per activity code across folds:")
        for activity, rmse in sorted(average_rmse_per_activity.items()):
            print(f"Activity {activity}: RMSE = {rmse:.4f}")
        activity_total_counts = {k: len(v) for k, v in activity_rmse_across_folds.items()}
        total_rmse_weighted = sum(np.sum(rmses) for rmses in activity_rmse_across_folds.values())
        total_count = sum(activity_total_counts.values())
        
        average_rmse_weighted = total_rmse_weighted / total_count
        print("average weighted:",average_rmse_weighted)
        print("._" * 10, end=".\n")
        return 

            
    
