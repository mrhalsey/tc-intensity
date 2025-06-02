import torch
import torch.nn as nn
import numpy as np


"""
This is my very first implementation of a Convolutional LSTM which is best used for Spatio-Temporal Data
    spatio: data that works in a space
    temporal: data over a time period

ConvLSTMs:
    these models take the timeseries aspects of memorization to recognize patterns from LSTMs and the very perceptive
    ability to understand patterns in images from convolutional neural networks but for numerical data instead

Structure:
    W: weight matrix
    X: data matrix
    H: network output
    C: network cell
    b: bias matrix
    o: dot product

    input gate = Sigmoid(W_x * X + W * H + W o C + b)
"""

class ConvLSTMCell(nn.Module):
    def __init__(self):
        pass
    def forward(self,x):
        pass

class ConvLSTM(nn.Module):
    def __init(self):
        pass
    def forward(self, x):
        pass
