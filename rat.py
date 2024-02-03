"""
RAT

This file contains all the necessary class and functions for the newer version of the mouse V1 project. This includes:

get_data: function which returns tuning curves from the data

MouseLossFunction: The loss function which incorporates the MMD loss for E and I neuron separately and the average step from running
                    euler to fixed point

Rodents: Parent class containing the hyperparameters and initialisation.

ConnectivityWeights: Inherit from Rodents for weight matrix generation.

NetworkExecuter: Inherit from Rodents for running the network given a weight matrix.

NOTE: THIS FILE DOES NOT INHERIT FROM torch.nn AND DOES NOT SUPPORT BACKPROPAGATION AND GRADIENT BASED OPTIMISATION.
        PLEASE REFER TO mouse.py FOR LEGACY CODE WHICH DOES SUPPORTS AUTOGRAD.

"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.set_default_dtype(torch.float32)

def get_data() -> list[torch.Tensor]:
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

        output.append(torch.tensor(result_array, device="cpu"))

    return output

class MouseLossFunction:
    def __init__(self, avg_step_weighting=0.002, high_contrast_index=7, device="cpu"):
        super().__init__()
        self.device = device
        self.one = torch.tensor(1)
        self.avg_step_weighting = avg_step_weighting
        self.high_contrast_index = high_contrast_index


    def calculate_loss(self, x_E: torch.Tensor, y_E: torch.Tensor, x_I: torch.Tensor, y_I: torch.Tensor, avg_step: torch.Tensor):
        E = self.MMD(self.centralise_all_curves(x_E), self.centralise_all_curves(y_E))
        I = self.MMD(self.centralise_all_curves(x_I), self.centralise_all_curves(y_I))  # TODO: Dont need to center the y_I and y_E all the time
        return E + I + (torch.maximum(self.one, avg_step) - 1) * self.avg_step_weighting, E + I

    
    def MMD(self, x: torch.Tensor, y: torch.Tensor):
        XX  = self.individual_terms_single_loop(x, x)
        XY  = self.individual_terms_single_loop(x, y)
        YY  = self.individual_terms_single_loop(y, y)
        return XX + YY - 2 * XY
    

    def individual_terms_single_loop(self, x: torch.Tensor, y: torch.Tensor):
        N = x.shape[0]
        M = y.shape[0]
        accum_output = torch.tensor(0, device=self.device)
        for i in range(N):
            x_repeated = x[i, :, :].unsqueeze(0).expand(M, -1, -1)
            accum_output = accum_output + torch.mean(self.kernel(y, x_repeated))
        return accum_output / N
    

    @staticmethod
    def kernel(x, y, w=1, axes=(-2, -1)):
        return torch.exp(-torch.sum((x - y) ** 2, dim=axes) / (2 * w**2))


    def get_max_index(self, tuning_curve):
        max_index = torch.argmax(tuning_curve[self.high_contrast_index])
        return max_index


    def centralise_curve(self, tuning_curve):
        max_index = self.get_max_index(tuning_curve)  # instead of max index, taking the mean might be better?
        shift_index = 6 - max_index  # 6 is used here as there are 13 orientations
        new_tuning_curve = torch.roll(tuning_curve, int(shift_index), dims=1)
        return new_tuning_curve


    def centralise_all_curves(self, responses):
        tuning_curves = []
        for tuning_curve in responses:
            tuning_curves.append(self.centralise_curve(tuning_curve))
        return torch.stack(tuning_curves)
    

class Rodents:
    pass


class ConnectivityWeights(Rodents):

    # Generate weight matrix

    # Check whether the weight matrix is valid

    pass


class NetworkExecuter(Rodents):

    # Update weight matrix

    # Run the network using euler and ricciardi

    pass