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

class PositionalEncoding(nn.Module):
                        def __init__(self, d_model, max_len=100):
                            super().__init__()
                            pe = torch.zeros(max_len, d_model)
                            position = torch.arange(0, max_len).unsqueeze(1)
                            div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
                            pe[:, 0::2] = torch.sin(position * div_term)
                            pe[:, 1::2] = torch.cos(position * div_term)
                            self.pe = pe.unsqueeze(0)
                    
                        def forward(self, x):
                            return x + self.pe[:, :x.size(1)].to(x.device)
                    
class MultiheadAttention1DModel(nn.Module):
                        def __init__(self, input_channels=1,kernel_size=3, d_model=64, nhead=8, num_layers=2, output_size=10):
                            super().__init__()
                            self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=kernel_size, padding=kernel_size//2)
                            self.pos_encoder = PositionalEncoding(d_model)
                            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True,dropout=0.1)
                            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                            self.output_head = nn.Sequential(
                                nn.Linear(d_model, 64),
                                nn.ReLU(),
                                nn.Linear(64, 1)
                            )
                            self.output_size = output_size
                    
                        def forward(self, x):
                            x = self.input_proj(x).permute(0, 2, 1)  # (B, L, C)
                            x = self.pos_encoder(x)
                            x = self.transformer_encoder(x)
                            x = self.output_head(x).squeeze(-1)  # (B, L)
                            return x
                    
                            
                            
def root_mean_squared_error(y_true, y_pred):
                        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))