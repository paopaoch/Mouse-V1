import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_data(device="cpu"):
    df = pd.read_csv("./data/K-Data.csv")
    v1 = df.query("region == 'V1'")
    m = v1.m.unique()[2]
    v1 = v1[v1.m == m]
    v1 = v1.copy()  # to prevent warning
    v1["mouse_unit"] = v1["m"] + "_" + v1["u"].astype(str)
    v1 = v1.groupby(["mouse_unit", "unit_type", "grat_orientation", "grat_contrast", "grat_spat_freq", "grat_phase"]).mean(numeric_only=True).reset_index()
    v1 = v1[["mouse_unit", "unit_type", "grat_orientation", "grat_contrast", "grat_spat_freq", "grat_phase", "response"]]


    excit = v1.query("unit_type == 'excit'")
    inhib = v1.query("unit_type == 'inhib'")
    output = []

    for response in [excit, inhib]:
        unique_units = response['mouse_unit'].unique()
        unique_orientation = response['grat_orientation'].unique()
        unique_contrast = response['grat_contrast'].unique()
        unique_spat_freq = response['grat_spat_freq'].unique()
        unique_phase = response['grat_phase'].unique()


        shape = (len(unique_units), len(unique_orientation), len(unique_contrast), len(unique_spat_freq), len(unique_phase))
        result_array = np.full(shape, np.nan)

        # Iterate through the DataFrame and fill the array
        for _, row in tqdm(response.iterrows()):
            u_index = np.where(unique_units == row['mouse_unit'])[0][0]
            orientation_index = np.where(unique_orientation == row['grat_orientation'])[0][0]
            contrast_index = np.where(unique_contrast == row['grat_contrast'])[0][0]
            spat_freq_index = np.where(unique_spat_freq == row['grat_spat_freq'])[0][0]
            phase_index = np.where(unique_phase == row['grat_phase'])[0][0]
            result_array[u_index, orientation_index, contrast_index, spat_freq_index, phase_index] = row['response']

        result_array = np.mean(np.mean(result_array, axis=4), axis=3)
        result_array = result_array.transpose((0, 2, 1))
        result_array = result_array * 1000

        output.append(torch.tensor(result_array, device=device, dtype=torch.float32))

    return output


class Rodents:
    def __init__(self, neuron_num, ratio=0.8, device="cpu", feed_forward_num=100):
        self.neuron_num = neuron_num
        neuron_num_e = int(neuron_num * ratio)
        neuron_num_i = neuron_num - neuron_num_e
        self.neuron_num_e = neuron_num_e
        self.neuron_num_i = neuron_num_i
        self.feed_forward_num = feed_forward_num
        self.device = device
        self.ratio = ratio

        self.pref_E = torch.linspace(0, 179.99, self.neuron_num_e, device=device, requires_grad=False)
        self.pref_I = torch.linspace(0, 179.99, self.neuron_num_i, device=device, requires_grad=False)
        self.pref = torch.cat([self.pref_E, self.pref_I]).to(device)

        self.pref_F = torch.linspace(0, 179.99, self.feed_forward_num, device=device, requires_grad=False)


    @staticmethod
    def _cric_gauss(x: torch.Tensor, w):
        """Circular Gaussian from 0 to 180 deg"""
        return torch.exp((torch.cos(x * torch.pi / 90) - 1) / (4 * torch.square(torch.pi / 180 * w)))


    @staticmethod
    def _sigmoid(array: torch.Tensor, steepness=1, scaling=1):
        """returns the sigmoidal value of the input tensor. The default steepness is 1."""
        return scaling / (1 + torch.exp(-steepness * array))
