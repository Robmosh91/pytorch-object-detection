import torch
import torch.nn as nn

from utils.utils import get_region_boxes, parse_config


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
        net_info = blocks.pop(0)
        modules = nn.ModuleList()
        # Number of previous filters is initialized to 3 since we're forwarding RGB images with 3 channels
        prev_filters = int(net_info['channels'])
        out_filters = []
        prev_stride = 1
        out_strides = []
        conv_id = 0
        for index, block in enumerate(blocks):
            if block['type'] == 'convolutional':
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
                prev_stride = stride * prev_stride
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
                module = Empty()
            elif block['type'] == 'shortcut':
                ind = len(modules)
                prev_filters = out_filters[ind - 1]
                prev_stride = out_strides[ind - 1]
                module = Empty()
            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                prev_stride = prev_stride // stride
                module = Upsample(stride)
            elif block['type'] == 'yolo':
                yolo_layer = Yolo()
                anchors = block['anchors'].split(',')
                anchor_mask = block['mask'].split(',')
                yolo_layer.anchor_mask = [int(i) for i in anchor_mask]
                yolo_layer.anchors = [float(i) for i in anchors]
                yolo_layer.num_classes = int(block['classes'])
                yolo_layer.num_anchors = int(block['num'])
                yolo_layer.anchor_step = len(yolo_layer.anchors) // yolo_layer.num_anchors
                yolo_layer.stride = prev_stride
                module = yolo_layer
            else:
                raise ValueError('Unknown block type %s not recognized.' % (block['type']))
            out_filters.append(prev_filters)
            out_strides.append(prev_stride)
            modules.append(module)
        return modules


class Empty(nn.Module):
    def __init__(self):
        super(Empty, self).__init__()

    def forward(self, x):
        return x


class Upsample(nn.Module):
    """
    Layer that Upsamples an Input of shape (B, C, H, W) to a given size.
    """
    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, stride, W, stride).contiguous().view(B, C, H * stride, W * stride)
        return x


class Yolo(nn.Module):
    def __init__(self, anchor_mask: list = [], num_classes: int = 0, anchors: list = [], num_anchors: int = 1):
        super(Yolo, self).__init__()
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) / num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = 32
        self.seen = 0

    def forward(self, output, nms_thresh):
        self.thresh = nms_thresh
        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m * self.anchor_step:(m + 1) * self.anchor_step]
        masked_anchors = [anchor / self.stride for anchor in masked_anchors]
        boxes = get_region_boxes(output.data, self.thresh, self.num_classes, masked_anchors, len(self.anchor_mask))
        return boxes
