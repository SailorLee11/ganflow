"""
@Time    : 2021/9/1 16:43
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : chrislistudy@163.com
-------------------------------------------------
@FileName: evalute_unsw.py
@Software: PyCharm
"""


import torch


def set_input( input: torch.Tensor):
    """ Set input and ground truth

    Args:
        input (FloatTensor): Input data for batch i.
    """
    with torch.no_grad():
        input.resize_(input[0].size()).copy_(input[0])
        gt.resize_(input[1].size()).copy_(input[1])
        label.resize_(input[1].size())

        # Copy the first batch as the fixed input.
        if self.total_steps == self.opt.batchsize:
            self.fixed_input.resize_(input[0].size()).copy_(input[0])

model = torch.load('../output/ganomaly/unsw/train/weights/netG.pth')

