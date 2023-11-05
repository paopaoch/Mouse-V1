import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.functional import F
from sim_utils import Phi, circ_gauss, Euler2fixedpt
from tqdm import tqdm


class MMDLossFunction(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, X, Y):
        XX = torch.mean(self.kernel(X[None, :, :, :], X[:, None, :, :]))
        XY = torch.mean(self.kernel(X[None, :, :, :], Y[:, None, :, :]))
        YY = torch.mean(self.kernel(Y[None, :, :, :], Y[:, None, :, :]))

        # XX.requires_grad_(True)
        # XY.requires_grad_(True)
        # YY.requires_grad_(True)
        output = XX - 2 * XY + YY
        output.requires_grad_(True)
        return output


    @staticmethod
    def kernel(x, y, w=1, axes=(-2, -1)):
        x, y = torch.tensor(x), torch.tensor(y)
        return torch.exp(-torch.sum((x - y) ** 2, dim=axes) / (2 * w**2))


class NeuroNN(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """

    def __init__(self, J_array: list, P_array: list, w_array: list, neuron_num: int, ratio=0.8, scaling_g=1, w_ff=30, sig_ext=5):
        # need error handling here for shape of parameters
        super().__init__()

        # initialize weights with input parameters
        hyperparameters = torch.tensor([J_array] + [P_array] + [w_array], requires_grad=True)

        # make hyperparameters torch parameters
        self.hyperparameters = nn.Parameter(hyperparameters)

        self.neuron_num = neuron_num
        neuron_num_e = int(neuron_num * ratio)
        neuron_num_i = neuron_num - neuron_num_e
        # neuron_types = np.array([1] * neuron_num_e + [-1] * neuron_num_i)
        self.neuron_num_e = neuron_num_e
        self.neuron_num_i = neuron_num_i
        # self.neuron_types = neuron_types

        pref_E = np.linspace(0, 179.99, neuron_num_e, False)
        pref_I = np.linspace(0, 179.99, neuron_num_i, False)
        self.pref = np.concatenate([pref_E, pref_I])
        self._generate_diff_thetas_matrix()

        # Global Parameters
        self.scaling_g = scaling_g * np.ones(neuron_num)
        self.w_ff = w_ff * np.ones(neuron_num)
        self.sig_ext = sig_ext * np.ones(neuron_num)
        T_alpha = 0.5
        T_E = 0.01
        T_I = 0.01 * T_alpha
        self.T = np.concatenate([T_E * np.ones(neuron_num_e), T_I * np.ones(neuron_num_i)])
        self.T_inv = np.reciprocal(self.T)

        # Membrane time constants for excitatory and inhibitory
        tau_alpha = 1
        tau_E = 0.01
        tau_I = 0.01 * tau_alpha
        # Membrane time constant vector for all cells
        self.tau = np.concatenate([tau_E * np.ones(neuron_num_e), tau_I * np.ones(neuron_num_i)])

        # Refractory periods for exitatory and inhibitory
        tau_ref_E = 0.005
        tau_ref_I = 0.001
        self.tau_ref = np.concatenate([tau_ref_E * np.ones(neuron_num_e), tau_ref_I * np.ones(neuron_num_i)])

        self.weights = None
        self.weights2 = None
        self.update_weight_matrix()

        # # Contrast and orientation ranges
        self.orientations = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
        # self.orientations = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165]
        self.contrasts = [0., 0.0432773, 0.103411, 0.186966, 0.303066, 0.464386, 0.68854, 1.]

        # plt.imshow(self.weights)
        # plt.colorbar()
        # plt.show()


    def forward(self):
        """Implementation of the forward step in pytorch.
        """
        print(self.hyperparameters.data)
        self.update_weight_matrix()
        tuning_curves = self.run_all_orientation_and_contrast()
        return tuning_curves


    # ------------------GET WEIGHT MATRIX--------------------------


    def update_weight_matrix(self) -> None:
        self.weights = self.generate_weight_matrix()
        self.weights2 = np.square(self.weights)


    def generate_weight_matrix(self) -> np.ndarray:
        # Calculate matrix relating to the hyperparameters and connection types
        connection_matrix = self._generate_connection_matrix() # Generate connection matrix randomly
        efficacy_matrix = self._generate_parameter_matrix(self.hyperparameters.data[0], connection_matrix)
        prob_matrix = self._generate_parameter_matrix(self.hyperparameters.data[1], connection_matrix)
        width_matrix = self._generate_parameter_matrix(self.hyperparameters.data[2], connection_matrix)
        self._generate_z_matrix(width=width_matrix)
        
        weight_matrix = efficacy_matrix * self._sigmoid(prob_matrix*self.z_matrix - np.random.rand(self.neuron_num, self.neuron_num))
        return weight_matrix


    def _generate_connection_matrix(self): # We can save the output of this function to a global var to prevent re running.
        connection_matrix = np.zeros((self.neuron_num, self.neuron_num), dtype=int)

        # Set values for EE connections
        connection_matrix[self.neuron_num_e:, self.neuron_num_e:] = 3
        # Set values for EI connections
        connection_matrix[:self.neuron_num_e, self.neuron_num_e:] = 2
        # Set values for IE connections
        connection_matrix[self.neuron_num_e:, :self.neuron_num_e] = 1
        # Set values for II connections
        connection_matrix[:self.neuron_num_e, :self.neuron_num_e] = 0

        return connection_matrix
    

    def _generate_parameter_matrix(self, params: torch.Tensor, connection_matrix: np.ndarray): # can combine
        params_matrix = params[connection_matrix.astype(int)]
        return np.array(params_matrix)
    

    def _generate_diff_thetas_matrix(self):
        output_orientations = np.tile(self.pref, (self.neuron_num, 1))
        input_orientations = output_orientations.T
        diff_orientations = np.abs(input_orientations - output_orientations)
        self.diff_orientations = diff_orientations
        return diff_orientations


    def _generate_z_matrix(self, width: np.ndarray):
        self.z_matrix = np.exp((np.cos(2 * np.pi/180 * self.diff_orientations) - 1) / (4 * (np.pi/180 * width)**2))
        return self.z_matrix


    @staticmethod
    def _sigmoid(array):
        return 1 / (1 + np.exp(-32*array)) 


    #---------------------RUN THE NETWORK TO GET STEADY STATE OUTPUT------------------------


    def get_steady_state_output(self, contrast, grating_orientations):
        input_mean, input_sd = self._stim_to_inputs(contrast, grating_orientations, self.pref)
        r_fp, avg_step = self._solve_fixed_point(input_mean, input_sd, )
        return r_fp, avg_step


    def _get_mu_sigma(self, weights_matrix, weights_2_matrix, rate, input_mean, input_sd, tau):
        # Find net input mean and variance given inputs
        mu = tau * (weights_matrix @ rate) + input_mean
        sigma = np.sqrt(tau * (weights_2_matrix @ rate) + input_sd**2)
        return mu, sigma
    

    def _stim_to_inputs(self, contrast, grating_orientations, preferred_orientations):
        '''Set the inputs based on the contrast and orientation of the stimulus'''
        # Distribute parameters over all neurons based on type
        input_mean = contrast * 20 * self.scaling_g * circ_gauss(grating_orientations - preferred_orientations, self.w_ff)
        input_sd = self.sig_ext
        return input_mean, input_sd


    def _solve_fixed_point(self, input_mean, input_sd): # tau_ref varies with E and I
        r_init = np.zeros(self.neuron_num)
        # Define the function to be solved for
        def drdt_func(rate):
            return self.T_inv * (Phi(*self._get_mu_sigma(self.weights, self.weights2, rate, input_mean, input_sd, self.tau), 
                                     self.tau, 
                                     tau_ref=self.tau_ref) - rate)
            
        # Solve using Euler
        r_fp, avg_step = Euler2fixedpt(drdt_func, r_init)
        return r_fp, avg_step

    
    # -------------------------RUN OUTPUT TO GET TUNING CURVES--------------------


    def run_all_orientation_and_contrast(self) -> np.ndarray: # Change this to numpy mathmul
        all_rates = []
        for contrast in self.contrasts:
            steady_states = []
            for orientation in self.orientations:
                rate, _ = self.get_steady_state_output(contrast, orientation)
                steady_states.append(rate)
            all_rates.append(steady_states)
        output = self._transform_all_rates_to_tuning_curves(np.array(all_rates))
        return output
    

    @staticmethod
    def _transform_all_rates_to_tuning_curves(all_rates: np.ndarray):
        return np.transpose(all_rates, (2, 0, 1))

# x = torch.arange(1000)
# # # Contrast and orientation ranges
# orientations = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
# orientations = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165]
# contrasts = [0., 0.0432773, 0.103411, 0.186966, 0.303066, 0.464386, 0.68854, 1.]

# contrasts = 0.186966
# orientations = 15

# nnn = NeuroNN(J_array, P_array, w_array, 100)
# tuning_curves = nnn.forward()

# plt.plot(orientations, tuning_curves[40][7])
# plt.plot(orientations, tuning_curves[40][6])
# plt.plot(orientations, tuning_curves[40][5])
# plt.plot(orientations, tuning_curves[40][4])
# plt.plot(orientations, tuning_curves[40][3])
# plt.plot(orientations, tuning_curves[40][2])
# plt.plot(orientations, tuning_curves[40][1])
# plt.plot(orientations, tuning_curves[40][0])
# plt.show()


# plt.plot(orientations, tuning_curves[95][7])
# plt.plot(orientations, tuning_curves[95][6])
# plt.plot(orientations, tuning_curves[95][5])
# plt.plot(orientations, tuning_curves[95][4])
# plt.plot(orientations, tuning_curves[95][3])
# plt.plot(orientations, tuning_curves[95][2])
# plt.plot(orientations, tuning_curves[95][1])
# plt.plot(orientations, tuning_curves[95][0])
# plt.show()


'''TICKETS

    1. Change np array operations to torch tensor operations for GPU
    2. Translate sim_utils to torch tensor operations
    3. Data exploration: units of trial average response

'''
# NOTE: Take the average of the spacial freq and phase in the data

# NOTE: Optogenetics mice

J_array = [1.99, 1.9, -1.01, -0.79]
P_array = [0.11, 0.11, 0.45, 0.45]
w_array = [32, 32, 32, 32]

model = NeuroNN(J_array, P_array, w_array, 100)

optimizer = optim.Adam(model.parameters(), lr=0.001)

def training_loop(model, optimizer, Y, n=2):
    "Training loop for torch model."

    loss_function = MMDLossFunction()
    model.train()

    losses = []
    for _ in range(n):
        optimizer.zero_grad()
        preds = model()
        loss = loss_function(preds, Y)
        print(loss)
        loss.backward()
        print(model.hyperparameters)
        optimizer.step()
        losses.append(loss)
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


loss = training_loop(model, optimizer, result_array)
print(loss)

# https://towardsdatascience.com/how-to-use-pytorch-as-a-general-optimizer-a91cbf72a7fb
