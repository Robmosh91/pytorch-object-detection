import torch
import torch.nn as nn

from utils.utils import parse_config


class Darknet(nn.Module):
    def __init__(self, configfile):
        """
        Class that builds the Darknet given by a configfile.
        :param configfile:
        """
        super(Darknet, self).__init__()
        self.blocks = parse_config(configfile)
