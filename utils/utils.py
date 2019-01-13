import numpy as np
import torch


def box_transform(box: np.array, in_format: str = 'xywh', out_format: str = 'xyxy'):
    allowed_formats = ['xywh', 'xyxy']
    assert in_format in allowed_formats
    assert out_format in allowed_formats
    assert box.shape[-1] == 4
    if in_format == out_format:
        return box
    if box.ndim == 1:
        box = np.expand_dims(box, 0)
    if in_format == 'xywh':
        if out_format == 'xyxy':
            xmin = box[0]-0.5*box[2]
            xmax = box[0]+0.5*box[2]
            ymin = box[1]-0.5*box[3]
            ymax = box[1]+0.5*box[3]
            box = [xmin, ymin, xmax, ymax]
    elif in_format == 'xyxy':
        if out_format == 'xywh':
            w = box[2]-box[0]
            h = box[3]-box[1]
            cx = box[0] + w//2
            cy = box[1] + h//2
            box = [cx, cy, w, h]
    return box.squeeze()


def iou_np(boxes_1: np.array, boxes_2: np.array, style='xywh'):
    """
    :param boxes_1: A 1D or 2D-Array with the boxes on the last dimension.
    :param boxes_2: A 1D or 2D-Array with the boxes on the last dimension.
    :param style: Till now only 'xywh' style is supported.
    :return:
    """
    assert style == 'xywh'
    assert boxes_1.shape == boxes_2.shape
    if boxes_1.ndim == 1:
        boxes_1 = np.expand_dims(boxes_1, 0)
    if boxes_2.ndim == 1:
        boxes_2 = np.expand_dims(boxes_2, 0)
    ious = np.zeros((boxes_1.shape[0]))
    lr = np.maximum(
        np.minimum(boxes_1[:, 0] + 0.5 * boxes_1[:, 2], boxes_2[:, 0] + 0.5 * boxes_2[:, 2]) -
        np.maximum(boxes_1[:, 0] - 0.5 * boxes_1[:, 2], boxes_2[:, 0] - 0.5 * boxes_2[:, 2]),
        0
    )
    if np.any(lr > 0):
        tb = np.maximum(
            np.minimum(boxes_1[:, 1] + 0.5 * boxes_1[:, 3], boxes_2[:, 1] + 0.5 * boxes_2[:, 3]) -
            np.maximum(boxes_1[:, 1] - 0.5 * boxes_1[:, 3], boxes_2[:, 1] - 0.5 * boxes_2[:, 3]),
            0
        )
        if np.any(tb > 0):
            intersections = tb * lr
            unions = boxes_1[:, 2] * boxes_1[:, 3] + boxes_2[:, 2] * boxes_2[:, 3] - intersections
            ious = intersections/unions
    return ious


def parse_config(configfile: str):
    """
    :param configfile: Takes a Path to a configuration file for the DarkNet.
    :return: list: Returns a list of blocks. Each blocks describes a block in the neural
        network to be built. Block is represented as a dictionary in the list
    """
    # Parse config file and read all lines which are not empty.
    with open(configfile, 'r') as f:
        lines = [x.strip() for x in f.read().split('\n') if len(x) > 0 and x[0] != '#']

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:  # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)  # add it the blocks list
                block = {}  # re-init the block
            block["type"] = line.strip(' []')
        else:
            key, value = line.split("=")
            block[key.strip()] = value.strip()
    blocks.append(block)

    return blocks
