import torch
import torch.nn as nn

class MHWBaselineCNN(nn.Module):
    """
    A baseline 2D Convolutional Neural Network for SST forecasting.
    Takes an input tensor of shape [Batch, Time_Channels, Height, Width] 
    and outputs a predicted anomaly map of shape [Batch, 1, Height, Width].
    """
    def __init__(self, in_channels=5):
        super(MHWBaselineCNN, self).__init__()
        
        # Feature extraction layers using 2D convolutions
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass: defines how data flows through the network layers.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x
