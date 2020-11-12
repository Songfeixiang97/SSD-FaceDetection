#!/usr/bin/env python
# coding: utf-8
from train import Train

T = Train(
    batch = 64,
    Class = 2,
    box_num = [5,4,4,4,4,4],
    lr = 0.001,
    load_pretrain = True,
    model = 1,
    lr_updata = False,
    aug = True,
    uda = True
)

T.train(epoch_num = 1000, step_one_epoch = 1000, save_frq = 50, evl_frq = 50)




