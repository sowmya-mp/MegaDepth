import math

import torch
import torch.nn.functional as F
from torch.nn import ReplicationPad3d
import torchvision

import inflate_utils
from network.layers.inception import inception
from network import hourglass as hourglass


class I3HourGlass(torch.nn.Module):
    def __init__(self, inceptionnet2d, frame_nb, inflate_block_convs=False):
        super(I3HourGlass, self).__init__()
        self.frame_nb = frame_nb
        self.inceptionnet3d = inflate_features(
            inceptionnet2d, inflate_block_convs=inflate_block_convs)

    def forward(self, inp):
        out = self.inceptionnet3d(inp)
        return out


class _Channel3d(torch.nn.Module):
    def __init__(self, channellayer2d, inflate_convs=False):
        super(_Channel3d, self).__init__()

        self.inflate_convs = inflate_convs
        self.list = torch.nn.ModuleList()
        self.block = []
        self.block1 = torch.nn.Sequential()
        self.block2 = torch.nn.Sequential()
        self.list.append(self.block1)
        self.list.append(self.block2)

        for name, child in channellayer2d.named_children():
            for nested_name, nested_child in child[0].named_children():
                if isinstance(nested_child, torch.nn.BatchNorm2d):
                    self.block1.add_module(nested_name, inflate_utils.inflate_batch_norm(nested_child))
                elif isinstance(nested_child, torch.nn.ReLU):
                    self.block1.add_module(nested_name, nested_child)
                elif isinstance(nested_child, torch.nn.Conv2d):
                    print('Here')
                    self.block1.add_module(nested_name, inflate_utils.inflate_conv(nested_child, 1))
                elif isinstance(nested_child, torch.nn.MaxPool2d) or isinstance(
                        nested_child, torch.nn.AvgPool2d):
                    self.block1.add_module(nested_name, inflate_utils.inflate_pool(nested_child))
                elif isinstance(nested_child, torch.nn.UpsamplingNearest2d):
                    print("Here")
                    self.block1.add_module(nested_name, inflate_utils.inflate_upsample(nested_child))
                elif isinstance(nested_child, inception):
                    self.block1.add_module(nested_name, inflate_utils.inception3d(nested_child, nested_child.input_size, nested_child.config))
                elif isinstance(nested_child, hourglass.Channels4) or isinstance(
                    nested_child, hourglass.Channels3) or isinstance(nested_child,
                    hourglass.Channels2) or isinstance(nested_child, hourglass.Channels1):
                    self.block1.add_module(nested_name, _Channel3d(nested_child, inflate_convs=self.inflate_convs))
                else:
                    raise ValueError(
                        '{} is not among handled layer types'.format(type(nested_child)))

            for nested_name, nested_child in child[1].named_children():
                if isinstance(nested_child, torch.nn.BatchNorm2d):
                    print('Here')
                    self.block2.add_module(nested_name, inflate_utils.inflate_batch_norm(nested_child))
                elif isinstance(nested_child, torch.nn.ReLU):
                    print('Here')
                    self.block2.add_module(nested_name, nested_child)
                elif isinstance(nested_child, torch.nn.Conv2d):
                    print('Here')
                    self.block2.add_module(nested_name, inflate_utils.inflate_conv(nested_child, 1))
                elif isinstance(nested_child, torch.nn.MaxPool2d) or isinstance(
                        nested_child, torch.nn.AvgPool2d):
                    print('Here')
                    self.block2.add_module(nested_name, inflate_utils.inflate_pool(nested_child))
                elif isinstance(nested_child, torch.nn.UpsamplingNearest2d):
                    print("Here")
                    self.block2.add_module(nested_name, inflate_utils.inflate_upsample(nested_child))
                elif isinstance(nested_child, inception):
                    print('Here inception')
                    self.block2.add_module(nested_name, inflate_utils.inception3d(nested_child, nested_child.input_size, nested_child.config))
                elif isinstance(nested_child, hourglass.Channels4) or isinstance(
                    nested_child, hourglass.Channels3) or isinstance(nested_child,
                    hourglass.Channels2) or isinstance(nested_child, hourglass.Channels1):
                    print('Here channel class')
                    self.block2.add_module(nested_name, _Channel3d(nested_child, inflate_convs=self.inflate_convs))
                else:
                    raise ValueError(
                        '{} is not among handled layer types'.format(type(nested_child)))


    def forward(self, x):
        return self.list[0](x) + self.list[1](x)


def inflate_features(inceptionnet2d, inflate_block_convs=False):
    """
    Inflates the feature extractor part of InceptionNet by adding the corresponding
    inflated modules and transfering the inflated weights
    """
    features3d = torch.nn.Sequential()
    for name, child in inceptionnet2d.named_children():
        if isinstance(child, torch.nn.Sequential):
            block = torch.nn.Sequential()
            for nested_name, nested_child in child.named_children():
                if isinstance(nested_child, torch.nn.BatchNorm2d):
                    block.add_module(nested_name, inflate_utils.inflate_batch_norm(nested_child))
                elif isinstance(nested_child, torch.nn.ReLU):
                    block.add_module(nested_name, nested_child)
                elif isinstance(nested_child, torch.nn.Conv2d):
                    block.add_module(nested_name, inflate_utils.inflate_conv(nested_child, time_dim=3))
                elif isinstance(nested_child, torch.nn.MaxPool2d) or isinstance(
                        nested_child, torch.nn.AvgPool2d):
                    block.add_module(nested_name, inflate_utils.inflate_pool(nested_child))
                elif isinstance(nested_child, hourglass.Channels4) or isinstance(
                    nested_child, hourglass.Channels3) or isinstance(nested_child,
                    hourglass.Channels2) or isinstance(nested_child, hourglass.Channels1):
                    block.add_module(nested_name, _Channel3d(nested_child, inflate_convs=inflate_block_convs))
                else:
                    raise ValueError(
                        '{} is not among handled layer types'.format(type(nested_child)))
            features3d.add_module(name, block)
        else:
            raise ValueError(
                '{} is not among handled layer types'.format(type(child)))
    return features3d
