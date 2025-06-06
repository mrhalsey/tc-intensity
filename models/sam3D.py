import torch
import torch.nn as nn
import numpy as np

"""
Inputs for the 3D module is a 5 dimensional tensor [batch, num_vars, depth, lat, lon]
"""
class SpatialAttention3D(nn.Module):
    def __init__(self):
        super(SpatialAttention3D, self).__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=2)
        self.avgpool = nn.AvgPool3d(kernel_size=2)

        # first set of convolutions
        self.conv3_a = nn.Conv3d(in_channels=64, out_channels=256, kernel_size=2)
        self.conv3_b = nn.Conv3d(in_channels=256, out_channels=64, kernel_size=2)

        #second set of convolutions
        self.conv4_a = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=2)
        self.conv4_b = nn.Conv3d(in_channels=256, out_channels=64, kernel_size=2)

        #activation function
        self.relu = nn.ReLU()
    
    def forward(self, x_3d: torch.Tensor) -> torch.Tensor:
        ### STAGE 1: Sequential Convolutions
        ##first stage to create the intermediate features
        m_out = self.maxpool(x_3d) ## maxpooling
        a_out = self.avgpool(x_3d) ## avgpooling
        combined_feats = torch.cat([m_out, a_out], dim=1)
        print(combined_feats.shape)

        #1st set of features
        stage1_int_feats = self.relu(self.conv3_a(combined_feats))
        stage1_out_feats = self.relu(self.conv3_b(stage1_int_feats))

        ## STAGE 2: Pooling and Convolutions
        m_out_s2 = self.maxpool(stage1_out_feats)
        a_out_s2 = self.avgpool(stage1_out_feats)

        ## 2nd set of features
        second_set_feats = torch.cat([m_out_s2, a_out_s2], dim=1)
        print(second_set_feats.shape)

        stage2_int_feats = self.relu(self.conv4_a(second_set_feats))
        stage2_out_feats = self.relu(self.conv4_b(stage2_int_feats))
        print(f"Stage 2 - stage2_out_feats shape: {stage2_out_feats.shape}")
        return stage2_out_feats