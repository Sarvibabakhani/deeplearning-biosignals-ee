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

class LSTMRegressor(nn.Module):
                    def __init__(self,input_size=1, num_time_steps=20):
                        super(LSTMRegressor, self).__init__()
                        self.num_time_steps = num_time_steps
                
                        self.lstm1 = nn.LSTM(input_size= input_size, hidden_size=128, batch_first=True)
                        self.dropout1 = nn.Dropout(0.1)
                
                        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
                        self.dropout2 = nn.Dropout(0.3)
                
                        # Flatten after second LSTM
                        self.flatten = nn.Flatten()
                
                        # Dense layers
                        self.dense1 = nn.Linear(64 * num_time_steps, 64)
                        self.dropout3 = nn.Dropout(0.2)
                        self.bn = nn.BatchNorm1d(64)
                
                        self.output_layer = nn.Linear(64, num_time_steps)  # Regression output
                
                    def forward(self, x):
                        # x: (batch, 1, time) → (batch, time, 1)
                        x = x.permute(0, 2, 1)
                
                        # LSTM layers
                        x, _ = self.lstm1(x)
                        x = self.dropout1(x)
                
                        x, _ = self.lstm2(x)
                        x = self.dropout2(x)
                
                        # Flatten
                        x = self.flatten(x)
                
                        # Dense + Dropout + BatchNorm
                        x = self.dense1(x)
                        x = self.dropout3(x)
                        x = self.bn(x)
                
                        # Output layer
                        return self.output_layer(x)
                    
                    
                            
def root_mean_squared_error(y_true, y_pred):
                return torch.sqrt(torch.mean((y_true - y_pred) ** 2))