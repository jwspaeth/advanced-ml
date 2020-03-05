
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

def val_learning_curve(dataset, model, exp_cfg, revived_cfg, results, filename):

    curve = results["history"]["val_loss"]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(curve)
    plt.title("Learning Curve")
    plt.legend()
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.xlabel("Epochs")

    fig.savefig("{}learning_curves.png".format(filename), dpi=fig.dpi)