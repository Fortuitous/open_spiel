"""
v12.3 Residual Tower Architecture for DMP-only Backgammon.
Inputs: 1220-float tensor (50 spatial planes + 20 global scalars).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ExpertEyesNet(nn.Module):
    def __init__(self, num_res_blocks=12, num_filters=256):
        super(ExpertEyesNet, self).__init__()
        
        # Initial Convolution: 50 planes -> Feature Space
        self.conv_in = nn.Conv2d(50, num_filters, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn_in = nn.BatchNorm2d(num_filters)
        
        # The Tower: 12 Residual Blocks
        self.res_tower = nn.Sequential(*[ResidualBlock(num_filters) for _ in range(num_res_blocks)])
        
        # Value Head (Win Probability)
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(24 + 20, 128)
        self.value_fc2 = nn.Linear(128, 1)
        
        # Policy Head (Move Selection)
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        # OpenSpiel's game.num_distinct_actions() = 913952
        self.policy_fc = nn.Linear(2 * 24 + 20, 913952)

    def forward(self, x_flat):
        # x_flat is [Batch, 1220]
        x_spatial = x_flat[:, :1200].view(-1, 50, 1, 24)
        x_scalar = x_flat[:, 1200:]
        
        # Spatial Processing
        x = F.relu(self.bn_in(self.conv_in(x_spatial)))
        x = self.res_tower(x)
        
        # Value Head (with Scalar Injection)
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 24)
        v_combined = torch.cat([v, x_scalar], dim=1)
        v_hidden = F.relu(self.value_fc1(v_combined))
        value = torch.tanh(self.value_fc2(v_hidden))
        
        # Policy Head (with Scalar Injection)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 2 * 24)
        p_combined = torch.cat([p, x_scalar], dim=1)
        policy = self.policy_fc(p_combined)
        
        return policy, value
