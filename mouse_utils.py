import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from sim_utils_torch import Phi, circ_gauss, Euler2fixedpt
from tqdm import tqdm


class MMDLossFunction(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, X, Y, avg_step):
        XX = torch.mean(self.kernel(X[None, :, :, :], X[:, None, :, :]))
        XY = torch.mean(self.kernel(X[None, :, :, :], Y[:, None, :, :]))
        YY = torch.mean(self.kernel(Y[None, :, :, :], Y[:, None, :, :]))

        output = XX - 2 * XY + YY + avg_step
        output.requires_grad_(True)
        return output


    @staticmethod
    def kernel(x, y, w=1, axes=(-2, -1)):
        if type(x) != torch.Tensor:
            x = torch.tensor(x)
        
        if type(y) != torch.Tensor:
            y = torch.tensor(y)

        return torch.exp(-torch.sum((x - y) ** 2, dim=axes) / (2 * w**2))


class NeuroNN(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """

    def __init__(self, J_array: list, P_array: list, w_array: list, neuron_num: int, ratio=0.8, scaling_g=1, w_ff=30, sig_ext=5):
        super().__init__()

        j_hyper = torch.tensor(J_array, requires_grad=True)
        self.j_hyperparameter = nn.Parameter(j_hyper)

        p_hyper = torch.tensor(P_array, requires_grad=True)
        self.p_hyperparameter = nn.Parameter(p_hyper)

        w_hyper = torch.tensor(w_array, requires_grad=True)
        self.w_hyperparameter = nn.Parameter(w_hyper)

        self.neuron_num = neuron_num
        neuron_num_e = int(neuron_num * ratio)
        neuron_num_i = neuron_num - neuron_num_e
        self.neuron_num_e = neuron_num_e
        self.neuron_num_i = neuron_num_i

        pref_E = torch.linspace(0, 179.99, neuron_num_e)
        pref_I = torch.linspace(0, 179.99, neuron_num_i)
        self.pref = torch.cat([pref_E, pref_I])
        self._generate_diff_thetas_matrix()

        # Global Parameters
        self.scaling_g = scaling_g * torch.ones(neuron_num)
        self.w_ff = w_ff * torch.ones(neuron_num)
        self.sig_ext = sig_ext * torch.ones(neuron_num)
        T_alpha = 0.5
        T_E = 0.01
        T_I = 0.01 * T_alpha
        self.T = torch.cat([T_E * torch.ones(neuron_num_e), T_I * torch.ones(neuron_num_i)])
        self.T_inv = torch.reciprocal(self.T)

        # Membrane time constants for excitatory and inhibitory
        tau_alpha = 1
        tau_E = 0.01
        tau_I = 0.01 * tau_alpha
        # Membrane time constant vector for all cells
        self.tau = torch.cat([tau_E * torch.ones(neuron_num_e), tau_I * torch.ones(neuron_num_i)])

        # Refractory periods for exitatory and inhibitory
        tau_ref_E = 0.005
        tau_ref_I = 0.001
        self.tau_ref = torch.cat([tau_ref_E * torch.ones(neuron_num_e), tau_ref_I * torch.ones(neuron_num_i)])

        self.weights = None
        self.weights2 = None
        self.update_weight_matrix()

        # # Contrast and orientation ranges
        self.orientations = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
        self.contrasts = [0., 0.0432773, 0.103411, 0.186966, 0.303066, 0.464386, 0.68854, 1.]


    def forward(self):
        """Implementation of the forward step in pytorch."""
        self.update_weight_matrix()
        tuning_curves, avg_step = self.run_all_orientation_and_contrast()
        return tuning_curves, avg_step


    # ------------------GET WEIGHT MATRIX--------------------------


    def update_weight_matrix(self) -> None:
        self.weights = self.generate_weight_matrix()
        self.weights2 = torch.square(self.weights)


    def generate_weight_matrix(self) -> torch.Tensor:
        # Calculate matrix relating to the hyperparameters and connection types
        connection_matrix = self._generate_connection_matrix()
        sign_matrix = self._generate_sign_matrix()
        p_hyperparameter = torch.abs(self.p_hyperparameter * torch.tensor([1, 1, 1, 1]))
        p_hyperparameter = p_hyperparameter / p_hyperparameter.sum()

        efficacy_matrix = self._generate_parameter_matrix(self.j_hyperparameter, connection_matrix)
        prob_matrix = self._generate_parameter_matrix(p_hyperparameter, connection_matrix)
        width_matrix = self._generate_parameter_matrix(self.w_hyperparameter, connection_matrix)
        self._generate_z_matrix(width=width_matrix)
        
        weight_matrix = sign_matrix * efficacy_matrix * self._sigmoid(prob_matrix*self.z_matrix - torch.rand(self.neuron_num, self.neuron_num))
        return weight_matrix


    def _generate_connection_matrix(self):
        connection_matrix = torch.zeros((self.neuron_num, self.neuron_num), dtype=torch.int32)

        # Set values for EE connections
        connection_matrix[self.neuron_num_e:, self.neuron_num_e:] = 3
        # Set values for EI connections
        connection_matrix[:self.neuron_num_e, self.neuron_num_e:] = 2
        # Set values for IE connections
        connection_matrix[self.neuron_num_e:, :self.neuron_num_e] = 1
        # Set values for II connections
        connection_matrix[:self.neuron_num_e, :self.neuron_num_e] = 0

        return connection_matrix
    

    def _generate_sign_matrix(self):
        sign_matrix = torch.zeros((self.neuron_num, self.neuron_num), dtype=torch.int32)

        # Set values for EE connections
        sign_matrix[self.neuron_num_e:, self.neuron_num_e:] = 1
        # Set values for EI connections
        sign_matrix[:self.neuron_num_e, self.neuron_num_e:] = 1
        # Set values for IE connections
        sign_matrix[self.neuron_num_e:, :self.neuron_num_e] = -1
        # Set values for II connections
        sign_matrix[:self.neuron_num_e, :self.neuron_num_e] = -1

        return sign_matrix
    

    def _generate_parameter_matrix(self, params: torch.Tensor, connection_matrix: torch.Tensor):
        connection_matrix = connection_matrix.type(torch.int64)
        params_matrix = params[connection_matrix]
        return params_matrix
    

    def _generate_diff_thetas_matrix(self):
        # output_orientations = np.tile(self.pref, (self.neuron_num, 1))
        output_orientations = self.pref.repeat(self.neuron_num, 1)
        input_orientations = output_orientations.T
        diff_orientations = torch.abs(input_orientations - output_orientations)
        self.diff_orientations = diff_orientations
        return diff_orientations


    def _generate_z_matrix(self, width: torch.Tensor):
        self.z_matrix = torch.exp((torch.cos(2 * torch.pi / 180 * self.diff_orientations) - 1) / (4 * (torch.pi / 180 * width)**2))
        return self.z_matrix


    @staticmethod
    def _sigmoid(array):
        return 1 / (1 + torch.exp(-32 * array))


    #---------------------RUN THE NETWORK TO GET STEADY STATE OUTPUT------------------------


    def get_steady_state_output(self, contrast, grating_orientations):
        input_mean, input_sd = self._stim_to_inputs(contrast, grating_orientations, self.pref)
        r_fp, avg_step = self._solve_fixed_point(input_mean, input_sd)
        return r_fp, avg_step


    def _get_mu_sigma(self, weights_matrix, weights_2_matrix, rate, input_mean, input_sd, tau):
        # Find net input mean and variance given inputs
        mu = tau * (weights_matrix @ rate) + input_mean
        sigma = torch.sqrt(tau * (weights_2_matrix @ rate) + input_sd**2)
        return mu, sigma
    

    def _stim_to_inputs(self, contrast, grating_orientations, preferred_orientations):
        '''Set the inputs based on the contrast and orientation of the stimulus'''
        # Distribute parameters over all neurons based on type
        # input_mean = circ_gauss(grating_orientations - preferred_orientations, self.w_ff)

        input_mean = contrast * 20 * self.scaling_g * circ_gauss(grating_orientations - preferred_orientations, self.w_ff)
        input_sd = self.sig_ext
        return input_mean, input_sd


    def _solve_fixed_point(self, input_mean, input_sd): # tau_ref varies with E and I
        r_init = torch.zeros(self.neuron_num) # Need to change this to a matrix
        # Define the function to be solved for
        def drdt_func(rate):
            return self.T_inv * (Phi(*self._get_mu_sigma(self.weights, self.weights2, rate, input_mean, input_sd, self.tau), 
                                     self.tau, 
                                     tau_ref=self.tau_ref) - rate)
            
        # Solve using Euler
        r_fp, avg_step = Euler2fixedpt(drdt_func, r_init)
        return r_fp, avg_step

    
    # -------------------------RUN OUTPUT TO GET TUNING CURVES--------------------


    def run_all_orientation_and_contrast(self) -> torch.Tensor: # Change this to numpy mathmul
        all_rates = torch.empty(0)
        avg_step_sum = torch.tensor(0)
        count = torch.tensor(0)
        for contrast in self.contrasts:
            steady_states = torch.empty(0)
            for orientation in self.orientations:
                rate, avg_step = self.get_steady_state_output(contrast, orientation)
                steady_states = torch.cat((steady_states, rate.unsqueeze(0)))
                avg_step_sum = avg_step_sum + avg_step
                count = count + torch.tensor(1)
            all_rates = torch.cat((all_rates, steady_states.unsqueeze(0)))
        output = all_rates.permute(2, 0, 1)
        return output, avg_step_sum / count


# NOTE: Optogenetics mice


def training_loop(model, optimizer, Y, n=220):
    "Training loop for torch model."

    loss_function = MMDLossFunction()
    model.train()

    losses = []
    for i in range(n):
        optimizer.zero_grad()
        preds, avg_step = model()
        loss = loss_function(preds, Y, avg_step)
        loss.backward()
        optimizer.step()
        losses.append(loss)
        print(f"ITTER: {i}", loss)
        print(model.j_hyperparameter)
        print(model.p_hyperparameter)
        print(model.w_hyperparameter)
        print("\n")

    return losses


df = pd.read_csv("./data/K-Data.csv")


v1 = df.query("region == 'V1'")
m = v1.m.unique()[2]
v1_data = v1[v1.m == m]
# v1_data = v1_data.query("grat_spat_freq == 0.332966").query("grat_phase == [180]")
v1_data = v1_data[['u', 'unit_type', 'grat_orientation', 'grat_contrast', 'response', 'smoothed_response']].reset_index(drop=True)
v1_data = v1_data.groupby(['unit_type','u', 'grat_contrast', 'grat_orientation'], as_index=False).mean()


# Get unique values for each column
unique_u = v1_data['u'].unique()
unique_contrast = v1_data['grat_contrast'].unique()
unique_orientation = v1_data['grat_orientation'].unique()


# Create a 3D numpy array filled with NaN values
shape = (len(unique_u), len(unique_contrast), len(unique_orientation))
result_array = np.full(shape, np.nan)


# Iterate through the DataFrame and fill the array
for index, row in v1_data.iterrows():
    u_index = np.where(unique_u == row['u'])[0][0]
    contrast_index = np.where(unique_contrast == row['grat_contrast'])[0][0]
    orientation_index = np.where(unique_orientation == row['grat_orientation'])[0][0]
    
    result_array[u_index, contrast_index, orientation_index] = row['response']

J_array = [1.99, 1.9, 1.01, 0.79]
P_array = [0.11, 0.11, 0.45, 0.45]
w_array = [32., 32., 32., 32.]

# J_array = [ 1.5343,  3.2766, -1.0630,  0.1529]
# P_array = [-0.3363,  1.4607,  1.6201, -0.2662]
# w_array = [32.4697, 33.3625, 33.0394, 31.8233]

model = NeuroNN(J_array, P_array, w_array, 169)

optimizer = optim.Adam(model.parameters(), lr=0.05)

loss = training_loop(model, optimizer, result_array)
print(loss)

# https://towardsdatascience.com/how-to-use-pytorch-as-a-general-optimizer-a91cbf72a7fb
