import torch
import torch.nn as nn


class Darknet(nn.Module):
    def __init__(self, configfile):
        """
        Class that builds the Darknet given by a configfile.
        :param configfile:
        """
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(configfile)