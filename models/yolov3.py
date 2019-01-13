import torch
import torch.nn as nn

from utils.utils import parse_config


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Darknet(nn.Module):
    def __init__(self, configfile):
        """
        Class that builds the Darknet given by a configfile.
        :param configfile:
        """
        super(Darknet, self).__init__()
        self.blocks = parse_config(configfile)
        self.model = self.create_model(blocks=self.blocks)

    def create_model(self, blocks: list):
        assert blocks[0]['type'] == 'net', \
            'First block must be of type net and should define the network input and hyperparameters.'
        modules = nn.ModuleList()
        prev_filters = 3
        out_filters = []
        prev_stride = 1
        out_strides = []
        conv_id = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id += 1
                batch_normalize = int(block.get('batch_normalize', '0'))
                bias = bool(1 - batch_normalize)
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size - 1) // 2 if is_pad else 0
                activation = block['activation']
                module = nn.Sequential()
                module.add_module('conv{0}'.format(conv_id),
                                  nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias))
                if batch_normalize:
                    module.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                if activation == 'leaky':
                    module.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                modules.append(module)
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(modules)
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                    prev_stride = out_strides[layers[0]]
                elif len(layers) == 2:
                    assert (layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                    prev_stride = out_strides[layers[0]]
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                modules.append(EmptyLayer())
            elif block['type'] == 'shortcut':

            elif block['type'] == 'upsample':

            elif block['type'] == 'yolo':



