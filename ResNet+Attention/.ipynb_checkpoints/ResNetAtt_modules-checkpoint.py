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

class NLBlock(nn.Module):
                        def __init__(self, num_channels):
                            super(NLBlock, self).__init__()
                            self.num_channels = num_channels
                            
                            # Equivalent to Theta, Phi, and G convolutions
                            self.theta = nn.Conv1d(num_channels, num_channels, kernel_size=1, stride=1, padding=0)
                            self.phi = nn.Conv1d(num_channels, num_channels, kernel_size=1, stride=1, padding=0)
                            self.g = nn.Conv1d(num_channels, num_channels, kernel_size=1, stride=1, padding=0)
                    
                        def forward(self, x):
                            batch_size, C, L = x.shape  # C = num_channels, L = sequence length
                    
                            # Apply convolutions
                            query = self.theta(x)  # (B, C, L)
                            key = self.phi(x)      # (B, C, L)
                            value = self.g(x)      # (B, C, L)
                    
                            # Compute attention scores
                            attention_scores = torch.bmm(query.permute(0, 2, 1), key)  # (B, L, L)
                            attention_scores = F.softmax(attention_scores / (C ** 0.5), dim=-1)  # Scale and normalize
                    
                            # Apply attention to value
                            attention_output = torch.bmm(attention_scores, value.permute(0, 2, 1))  # (B, L, C)
                            attention_output = attention_output.permute(0, 2, 1)  # Back to (B, C, L)
                    
                            # Residual connection
                            return x + attention_output

class ResidualBlock(nn.Module):
                        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=False):
                            super(ResidualBlock, self).__init__()
                            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
                            self.bn1 = nn.BatchNorm1d(out_channels)
                            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
                            self.bn2 = nn.BatchNorm1d(out_channels)
                            self.downsample = downsample
                            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride) if downsample else nn.Identity()
                    
                        def forward(self, x):
                            residual = self.shortcut(x)
                            out = torch.relu(self.bn1(self.conv1(x)))
                            out = self.bn2(self.conv2(out))
                            out += residual
                            return torch.relu(out)
class ResNet1D_Att(nn.Module):
                        def __init__(self, input_channels=1, output_size=10):
                            super(ResNet1D_Att, self).__init__()
                            
                            self.initial_conv = nn.Sequential(
                                nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
                            )
                            
                            self.res_layers = nn.Sequential(
                                ResidualBlock(64, 128, downsample=True),
                                ResidualBlock(128, 256, downsample=True),
                                ResidualBlock(256, 512, downsample=True)
                            )
                            
                            self.non_local_block = NLBlock(512)  # Add Non-Local Block
                            
                            self.fc = nn.Linear(512, output_size)
                    
                        def forward(self, x):
                            x = self.initial_conv(x)
                            x = self.res_layers(x)
                            x = self.non_local_block(x)  # Apply NLBlock
                            x = torch.mean(x, dim=2)  # Global Average Pooling
                            x = self.fc(x)
                            return x

def root_mean_squared_error(y_true, y_pred):
                        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))