import torch
from torch.utils.data import Dataset
import numpy as np

def find_gesture_onsets(gesture_labels):
    onsets = []
    for i in range(1, len(gesture_labels)):
        if gesture_labels[i] != 0 and gesture_labels[i-1] == 0:
            onsets.append(i)
    return onsets


def compute_time_to_gesture(gesture_labels, max_value=200):
    T = len(gesture_labels)
    onsets = find_gesture_onsets(gesture_labels)

    time_to = np.full(T, max_value, dtype=np.int32)

    for i in range(T):
        future_onsets = [o for o in onsets if o >= i]
        if future_onsets:
            time_to[i] = future_onsets[0] - i

    return time_to

def expand_gesture_labels(labels, back=5, forward=15):
    labels = labels.copy()

    for i in range(len(labels)):
        if labels[i] != 0:  # gesture event
            start = max(0, i - back)
            end = min(len(labels), i + forward)

            labels[start:end] = labels[i]

    return labels

class TimeToGestureDataset(Dataset):
    def __init__(self, imu_data, gesture_labels, window_size=24,stride=5, max_time=200):
        self.data = imu_data
        self.labels = expand_gesture_labels(gesture_labels, back=5, forward=15)
        self.window_size = window_size
        self.max_time = max_time
        self.stride = stride 

        time_to = compute_time_to_gesture(self.labels, max_value=self.max_time)
        # self.time_to = np.log1p(time_to) / np.log1p(self.max_time)
        # self.time_to = time_to / self.max_time
        tau = 20  
        self.time_to = np.exp(-time_to / tau)
        
        self.samples = []
        self.build_samples()

    def build_samples(self):
        T = len(self.data)

        for end in range(self.window_size, T - 1, self.stride):
            start = end - self.window_size

            x_window = self.data[start:end]
            y_target = self.time_to[end]

            self.samples.append((x_window, y_target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y