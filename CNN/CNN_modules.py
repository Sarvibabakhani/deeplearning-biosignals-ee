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


class CNNModel(nn.Module):
    def __init__(self, input_channels=1, num_time_steps=20, output_size=20):
        super(CNNModel, self).__init__()

        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 32, kernel_size=3, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Dropout(0.1),

            nn.Conv1d(32, 16, kernel_size=3, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Dropout(0.3)
        )

       #
        L_final = num_time_steps
        for _ in range(3):      # three pooling layers
            L_final //= 2       # floor division = correct pooling behavior
        #if we directly divide it by 8, for num_time_steps < 8, it does not work correctly

        flattened_dim = 16 * L_final

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 40),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(40, output_size)
        )

    def forward(self, x):
        """
        x shape: (batch_size, input_channels, num_time_steps)
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def root_mean_squared_error(y_true, y_pred):
                        return torch.sqrt(torch.mean((y_true - y_pred) ** 2))