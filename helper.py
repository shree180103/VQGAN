import torch 
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.group_norm=nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.group_norm(x)
    

class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block= nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1),
            GroupNorm(out_channels),
            swish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1, padding=1),
            GroupNorm(out_channels)
        )

    #  if  the no. of out channels and in channels are not equal, we need to add a conv layer to match the dimensions
        if in_channels != out_channels:
            self.match_dimensions = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.match_dimensions(x)+self.block(x)
        
        else:
            return x + self.block(x)
        

class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpsampleBlock, self).__init__()
        self.conv= nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x=F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
    
class DownsampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownsampleBlock, self).__init__()
        self.conv= nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(x)
    

class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.gn= GroupNorm(channels)

        self.q= nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.k= nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.v= nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

        self.project= nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
       

    def forward(self, x):

        norm= self.gn(x)
        q= self.q(norm)
        k=self.k(norm)
        v=self.v(norm)
        b, c, h, w = q.shape
        q = q.view(b, c, -1)
        k = k.view(b, c, -1).permute(0, 2, 1)
        v = v.view(b, c, -1)
        # Calculate attention scores
        attn_scores = (q @ k)/ (c ** 0.5)
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_output = attn_scores @ v
        attn_output = attn_output.view(b, c, h, w)

        A= self.project(attn_output)
        
       
        return x+A
    