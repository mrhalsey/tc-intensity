"""
SAM stands for Spatial Attention Model:
    this model is utilized to capture the spatial relationship between variables.

    Multidimensional factors determining typhoon intensity are taken into account, and
    several feature extraction appraches are utilized for various dimensional data

    the typhoon intensity environmental field is input to the SAM, and 2DConv is used to extract
    spatial features from 2D data such as SST and REL_HUM,
    where as 3DConv is used to extract features from 3D data
"""


"""
u10, v10:
    test on model and visualize output features
compare before and after images

mainly worry about 2D for now:
    over surface
"""

import torch
import torch.nn as nn
import numpy as np
from Calculations import normalize
import xarray as xr

#data = xr.open_dataset('../data/florence_2018.nc')
#sst = data['sst'].values
#sst = sst.to_numpy()
#tensor = torch.from_numpy(sst)

class SpatialAttention2D(nn.Module):
    def __init__(self):
        super(SpatialAttention2D, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.conv = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        
        m_out = self.maxpool(x)
        a_out = self.avgpool(x)

        combined_feats = torch.add(m_out, a_out)
        print(combined_feats.shape)
        first_conv = self.conv(combined_feats)

        mp2_feats = self.maxpool(first_conv)
        avp2_feats = self.avgpool(first_conv)

        feats_two = torch.add(mp2_feats, avp2_feats)
        print(feats_two.shape)

        final = self.conv2(feats_two)
        print(final.shape)
        return final
    

model = SpatialAttention2D()
test = torch.rand(1, 32, 64, 64)
output = model(test)
print("Output", output)