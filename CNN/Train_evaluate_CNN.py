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
from collections import defaultdict

from CNN_modules import CNNModel, root_mean_squared_error


def train_model(model, train_loader, val_loader,device,optimizer,criterion,scheduler,early_stopping_patience = 40, epochs=180):
                        train_losses, val_losses = [], []
                        best_val_loss = float('inf')
                        patience_counter = 0
                        best_state = None

                    
                        for epoch in range(epochs):
                            model.train()
                            train_loss = 0.0
                            for inputs, targets in train_loader:
                                inputs, targets = inputs.to(device), targets.to(device)
                                optimizer.zero_grad()
                                outputs = model(inputs)
                                loss = criterion(outputs, targets)
                                loss.backward()
                                optimizer.step()
                                train_loss += loss.item()
                            train_losses.append(train_loss / len(train_loader))
                    
                            # Validation
                            model.eval()
                            val_loss = 0.0
                            with torch.no_grad():
                                for inputs, targets in val_loader:
                                    inputs, targets = inputs.to(device), targets.to(device)
                                    outputs = model(inputs)
                                    val_loss += criterion(outputs, targets).item()
                            val_losses.append(val_loss / len(val_loader))
                            scheduler.step(val_losses[-1])
             
                    
                            if val_losses[-1] < best_val_loss:
                                best_val_loss = val_losses[-1]
                                patience_counter = 0
                                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                            else:
                                patience_counter += 1
                                if patience_counter >= early_stopping_patience:
                                    # print("Early stopping triggered.")
                                    break
                    
                        if best_state is not None:
                                    model.load_state_dict(best_state)

                        return model 

    
def evaluate_model(model, test_loader,device,criterion):
                        model.eval()
                        test_loss = 0.0
                        with torch.no_grad():
                            for inputs, targets in test_loader:
                                inputs, targets = inputs.to(device), targets.to(device)
                                outputs = model(inputs)
                                test_loss += criterion(outputs, targets).item()
                        
                        # print(f"Test Loss: {test_loss / len(test_loader):.4f}")
                        return test_loss / len(test_loader)


def evaluate_model_by_activity (model, test_df,device,criterion, scaler,sig= ["Interpolated Values_Waist Acceleration"], batch_size=8 , num_time_steps=20, activity_column='Activity Code',activity_rmse_across_folds = None):
                        model.eval()
                        
                        
                        activity_codes = test_df[activity_column].unique()
                    
                        for activity in activity_codes:
                            df_activity = test_df[test_df[activity_column] == activity]
                    
                            X_test = df_activity[sig].values
                            y_test = df_activity[['Interpolated Values_Enregy expendiures']].values
                    
                            # Drop samples that can't form full time_steps window
                            num_test_samples = len(X_test) // num_time_steps * num_time_steps
                            if num_test_samples == 0:
                                continue
                            X_test = X_test[:num_test_samples].reshape(-1,len(sig), num_time_steps)
                            y_test = y_test[:num_test_samples].reshape(-1, num_time_steps)
                             
                    
                            X_test_reshaped = X_test.transpose(0, 2, 1).reshape(-1, len(sig))
                            X_test_scaled = scaler.transform(X_test_reshaped).reshape(-1, num_time_steps, len(sig)).transpose(0, 2, 1)
                    
                            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
                            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
                            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
                            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                            test_loss = 0.0
                            total_samples = 0
                            
                            with torch.no_grad():
                                for inputs, targets in test_loader:
                                    inputs, targets = inputs.to(device), targets.to(device)
                                    outputs = model(inputs)
                                    batch_rmse = criterion(outputs, targets).item()
                                    batch_s = targets.size(0)  # number of sequences in this batch
                                    test_loss += batch_rmse * batch_s
                                    total_samples += batch_s
                            
                            average_rmse = test_loss / total_samples
                            activity_rmse_across_folds[activity].append(average_rmse)
                        return activity_rmse_across_folds

