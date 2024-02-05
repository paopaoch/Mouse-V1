import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import time
from scipy.stats import linregress
from tqdm import tqdm
from rat import get_data

def print_tuning_curve(tuning_curve):
    if type(tuning_curve) == torch.Tensor:
        tuning_curve = np.array(tuning_curve)

    plt.imshow(tuning_curve, cmap='viridis')
    plt.colorbar()
    plt.title("I Neuron (Pre-trained Model)")
    plt.xlabel("orientation index")
    plt.ylabel("contrast index")
    plt.show()

    for c in tuning_curve:
        plt.plot(c)

    plt.show()


def print_activity(responses):
    one_res = []
    for tuning_curve in responses:
        one_res.append(tuning_curve[7][4])

    plt.plot(one_res)
    plt.title("Activity of the network")
    plt.xlabel("Neuron Index")
    plt.ylabel("Response / Hz")
    plt.show()


