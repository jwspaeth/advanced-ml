#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from config.config_handler import config_handler
from supercomputer import train
from datasets import Core50Dataset

if __name__ == "__main__":
    dataset = Core50Dataset()
    data_dict = dataset.load_data()

    scissor_inds = []
    mug_inds = []
    for i in range(data_dict["train"]["outs"].shape[0]):

        if data_dict["train"]["outs"][i][0] == 1:
            scissor_inds.append(i)
        else:
            mug_inds.append(i)
    print("Num train scissors: {}".format(len(scissor_inds)))
    print("Num train mug_inds: {}".format(len(mug_inds)))

    print("TRAIN SCISSORS")
    for ind in scissor_inds[:5]:
        plt.imshow(data_dict["train"]["ins"][ind].astype(int))
        print("Train outs {}: {}".format(ind, data_dict["train"]["outs"][ind]))
        plt.show()

    print('TRAIN MUGS')
    for ind in mug_inds[:5]:
        plt.imshow(data_dict["train"]["ins"][ind].astype(int))
        print("Train outs {}: {}".format(ind, data_dict["train"]["outs"][ind]))
        plt.show()
    print()

    scissor_inds = []
    mug_inds = []
    for i in range(data_dict["val"]["outs"].shape[0]):

        if data_dict["val"]["outs"][i][0] == 1:
            scissor_inds.append(i)
        else:
            mug_inds.append(i)
    print("Num val scissors: {}".format(len(scissor_inds)))
    print("Num val mug_inds: {}".format(len(mug_inds)))

    print("VAL SCISSORS")
    for ind in scissor_inds[:5]:
        plt.imshow(data_dict["val"]["ins"][ind].astype(int))
        print("Val outs {}: {}".format(ind, data_dict["val"]["outs"][ind]))
        plt.show()

    print('VAL MUGS')
    for ind in mug_inds[:5]:
        plt.imshow(data_dict["val"]["ins"][ind].astype(int))
        print("Val outs {}: {}".format(ind, data_dict["val"]["outs"][ind]))
        plt.show()
    print()