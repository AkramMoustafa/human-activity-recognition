# src/models/gesture_soon_classifier.py

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        out = h_n[-1]   # last layer hidden state
        out = self.fc(out)
        return out  # NO sigmoid here