import torch
from network.layers.inception import inception

def inflate_conv(conv2d,
                 time_dim=3,
                 time_padding=0,
                 time_stride=1,
                 time_dilation=1,
                 center=False):
    # To preserve activations, padding should be by continuity and not zero
    # or no padding in time dimension
    kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
    padding = (time_padding, conv2d.padding[0], conv2d.padding[1])
    stride = (time_stride, conv2d.stride[0], conv2d.stride[0])
    dilation = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])
    conv3d = torch.nn.Conv3d(
        conv2d.in_channels,
        conv2d.out_channels,
        kernel_dim,
        padding=padding,
        dilation=dilation,
        stride=stride)
    # Repeat filter time_dim times along time dimension
    weight_2d = conv2d.weight.data
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim

    # Assign new params
    conv3d.weight = torch.nn.Parameter(weight_3d)
    conv3d.bias = conv2d.bias
    return conv3d


def inflate_batch_norm(batch2d):
    # In pytorch 0.2.0 the 2d and 3d versions of batch norm
    # work identically except for the check that verifies the
    # input dimensions

    batch3d = torch.nn.BatchNorm3d(batch2d.num_features)
    # retrieve 3d _check_input_dim function
    batch2d._check_input_dim = batch3d._check_input_dim
    return batch3d


def inflate_pool(pool2d,
                 time_dim=1,
                 time_padding=0,
                 time_stride=None,
                 time_dilation=1):
    kernel_dim = (time_dim, pool2d.kernel_size, pool2d.kernel_size)
    padding = (time_padding, pool2d.padding, pool2d.padding)
    if time_stride is None:
        time_stride = time_dim
    stride = (time_stride, pool2d.stride, pool2d.stride)
    if isinstance(pool2d, torch.nn.MaxPool2d):
        dilation = (time_dilation, pool2d.dilation, pool2d.dilation)
        pool3d = torch.nn.MaxPool3d(
            kernel_dim,
            padding=padding,
            dilation=dilation,
            stride=stride,
            ceil_mode=pool2d.ceil_mode)
    elif isinstance(pool2d, torch.nn.AvgPool2d):
        pool3d = torch.nn.AvgPool3d(kernel_dim, stride=stride)
    else:
        raise ValueError(
            '{} is not among known pooling classes'.format(type(pool2d)))
    return pool3d


def inflate_upsample(upsample2d,
                     time_dim=1):
    scale_factor_dim = (time_dim, upsample2d.scale_factor, upsample2d.scale_factor)
    upsample3d = torch.nn.Upsample(scale_factor=scale_factor_dim, mode='nearest')
    return upsample3d


class inception3d(torch.nn.Module):
    def __init__(self, inception, input_size, config):
        self.config = config
        super(inception3d, self).__init__()
        self.convs = torch.nn.ModuleList()

        # Base 1*1 conv layer
        self.convs.append(torch.nn.Sequential(
            #torch.nn.Conv3d(input_size, config[0][0], 1),
            inflate_conv(inception.convs[0][0], time_dim=1),
            torch.nn.BatchNorm3d(config[0][0], affine=False),
            torch.nn.ReLU(True),
        ))

        # Additional layers
        for i in range(1, len(config)):
            filter = config[i][0]
            pad = int((filter-1) / 2)
            out_a = config[i][1]
            out_b = config[i][2]
            conv = torch.nn.Sequential(
                #torch.nn.Conv3d(input_size, out_a, 1),
                inflate_conv(inception.convs[i][0], time_dim=1),
                torch.nn.BatchNorm3d(out_a, affine=False),
                torch.nn.ReLU(True),
                #torch.nn.Conv3d(out_a, out_b, filter, padding=pad),
                inflate_conv(inception.convs[i][3], time_dim=filter, time_padding=pad),
                torch.nn.BatchNorm3d(out_b, affine=False),
                torch.nn.ReLU(True)
                )
            self.convs.append(conv)

    def __repr__(self):
        return "inception3d" + str(self.config)

    def forward(self, x):
        ret = []
        for conv in (self.convs):
            ret.append(conv(x))
        return torch.cat(ret,dim=1)