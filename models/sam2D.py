import torch
import torch.nn as nn
import numpy as np

"""
Inputs for the 2D Module is a 4 dimensional tensor [batch, num_vars, lat, lon]
"""
class SpatialAttention2D(nn.Module):
    def __init__(self, input_channels: int):
        super(SpatialAttention2D, self).__init__()

        self.input_channels = input_channels

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2)

        # first set of convolutions
        ## for the in and out features the first is the amount of variables you are inputing into the model
        ## the out is the amount of features you want the model to discover from the data
        self.conv1_a = nn.Conv2d(in_channels=input_channels * 2, out_channels=32, kernel_size=2, stride=1) 
        self.conv1_b = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1)

        #second set of convolutions
        self.conv2_a = nn.Conv2d(in_channels=64 * 2, out_channels=128, kernel_size=2, stride=1)
        self.conv2_b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1)

        #activation function
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ### STAGE 1: Sequential Convolutions
        ##first stage to create the intermediate features
        m_out = self.maxpool(x) ## maxpooling
        a_out = self.avgpool(x) ## avgpooling
        combined_feats = torch.cat([m_out, a_out], dim=1)
        print(combined_feats.shape)

        #1st set of features
        stage1_int_feats = self.relu(self.conv1_a(combined_feats))
        stage1_out_feats = self.relu(self.conv1_b(stage1_int_feats))

        ## STAGE 2: Pooling and Convolutions
        m_out_s2 = self.maxpool(stage1_out_feats)
        a_out_s2 = self.avgpool(stage1_out_feats)

        ## 2nd set of features
        second_set_feats = torch.cat([m_out_s2, a_out_s2], dim=1)
        print(second_set_feats.shape)

        stage2_int_feats = self.relu(self.conv2_a(second_set_feats))
        stage2_out_feats = self.relu(self.conv2_b(stage2_int_feats))
        print(f"Stage 2 - stage2_out_feats shape: {stage2_out_feats.shape}")
        return stage2_out_feats
    

model = SpatialAttention2D(input_channels=52)
test_tensor = torch.rand(44, 52, 141, 281)
output = model(test_tensor)
print("Output", output.shape)