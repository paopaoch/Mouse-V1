import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from sim_utils_torch import Phi, circ_gauss, Euler2fixedpt
from tqdm import tqdm


class MMDLossFunction(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device


    def forward(self, X, Y, avg_step):
        XX = torch.mean(self.kernel(X[None, :, :, :], X[:, None, :, :]))
        XY = torch.mean(self.kernel(X[None, :, :, :], Y[:, None, :, :]))
        YY = torch.mean(self.kernel(Y[None, :, :, :], Y[:, None, :, :]))

        output = XX - 2 * XY + YY + avg_step * 0.002
        output.requires_grad_(True)
        return output


    def kernel(self, x, y, w=1, axes=(-2, -1)):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, device=self.device)
        
        if type(y) != torch.Tensor:
            y = torch.tensor(y, device=self.device)

        return torch.exp(-torch.sum((x - y) ** 2, dim=axes) / (2 * w**2))


class NeuroNN(nn.Module):
    """
    ### This class wraps the logic for the forward pass for modelling the mouse V1

    The forward pass performs two computations:
    1. Update the weight matrix
    2. Solves for fixed point at all contrast and orientation combinations
    """

    def __init__(self, J_array: list, P_array: list, w_array: list, neuron_num: int, ratio=0.8, scaling_g=1, w_ff=30, sig_ext=5, device="cpu"):
        super().__init__()
        self.device = device

        j_hyper = torch.tensor(J_array, requires_grad=True, device=device)
        self.j_hyperparameter = nn.Parameter(j_hyper)

        p_hyper = torch.tensor(P_array, requires_grad=True, device=device)
        self.p_hyperparameter = nn.Parameter(p_hyper)

        w_hyper = torch.tensor(w_array, requires_grad=True, device=device)
        self.w_hyperparameter = nn.Parameter(w_hyper)

        self.neuron_num = neuron_num
        neuron_num_e = int(neuron_num * ratio)
        neuron_num_i = neuron_num - neuron_num_e
        self.neuron_num_e = neuron_num_e
        self.neuron_num_i = neuron_num_i

        pref_E = torch.linspace(0, 179.99, neuron_num_e, device=device)
        pref_I = torch.linspace(0, 179.99, neuron_num_i, device=device)
        self.pref = torch.cat([pref_E, pref_I])
        self._generate_diff_thetas_matrix()

        # Global Parameters
        self.scaling_g = scaling_g * torch.ones(neuron_num, device=device)
        self.w_ff = w_ff * torch.ones(neuron_num, device=device)
        self.sig_ext = sig_ext * torch.ones(neuron_num, device=device)
        
        T_alpha = 0.5
        T_E = 0.01
        T_I = 0.01 * T_alpha
        self.T = torch.cat([T_E * torch.ones(neuron_num_e, device=device), T_I * torch.ones(neuron_num_i, device=device)])
        self.T_inv = torch.reciprocal(self.T)

        # Membrane time constants for excitatory and inhibitory
        tau_alpha = 1
        tau_E = 0.01
        tau_I = 0.01 * tau_alpha
        # Membrane time constant vector for all cells
        self.tau = torch.cat([tau_E * torch.ones(neuron_num_e, device=device), tau_I * torch.ones(neuron_num_i, device=device)])

        # Refractory periods for exitatory and inhibitory
        tau_ref_E = 0.005
        tau_ref_I = 0.001
        self.tau_ref = torch.cat([tau_ref_E * torch.ones(neuron_num_e, device=device), tau_ref_I * torch.ones(neuron_num_i, device=device)])

        self.connection_matrix = self._generate_connection_matrix()
        self.sign_matrix = self._generate_sign_matrix()

        self.weights = None
        self.weights2 = None
        self.update_weight_matrix()

        # # Contrast and orientation ranges
        self.orientations = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
        self.contrasts = [0., 0.0432773, 0.103411, 0.186966, 0.303066, 0.464386, 0.68854, 1.]

        # plt.imshow(self.weights.data, cmap="seismic", vmin=-np.max(np.abs(np.array(self.weights.data))), vmax=np.max(np.abs(np.array(self.weights.data))))
        # plt.colorbar()
        # plt.title("Connection weight matrix for 2000 neurons")
        # plt.xlabel("Neuron index")
        # plt.ylabel("Neuron index")
        # plt.show()
        
        # plt.savefig("./plots/weights_example_2000")

    def forward(self):
        """
        Implementation of the forward step in pytorch. 
        First updates the weights then solves the fix point for all orientation and contrasts.
        """
        self.update_weight_matrix()
        tuning_curves, avg_step = self.run_all_orientation_and_contrast()
        return tuning_curves, avg_step


    # ------------------GET WEIGHT MATRIX--------------------------


    def update_weight_matrix(self) -> None:
        """Update self.weights and self.weights2 which is the weights and weights squares respectively."""
        self.weights = self.generate_weight_matrix().to(self.device)
        self.weights2 = torch.square(self.weights)


    def generate_weight_matrix(self) -> torch.Tensor:
        """
        Method to generate a random weight matrix connection from the parameters J, P, w.
        
        Steps:
        1. constraints the parameters to be positive and the probability values to be between 0 and 1.
        2. transform the parameters to a matrix of size self.neuron_num by self.neuron_num for matrix arithmetics
        3. using the parameters matrix, calculate the weight matrix then adjust the signs of the matrix to obey Dales's law.
        """
        j_hyperparameter = torch.exp(self.j_hyperparameter * torch.tensor([1, 1, 1, 1], device=self.device))
        p_hyperparameter = torch.exp(self.p_hyperparameter * torch.tensor([1, 1, 1, 1], device=self.device))
        p_hyperparameter = self._sigmoid(p_hyperparameter, 2)
        w_hyperparameter = torch.exp(self.w_hyperparameter * torch.tensor([1, 1, 1, 1], device=self.device))

        efficacy_matrix = self._generate_parameter_matrix(j_hyperparameter)
        prob_matrix = self._generate_parameter_matrix(p_hyperparameter)
        width_matrix = self._generate_parameter_matrix(w_hyperparameter)
        self._generate_z_matrix(width=width_matrix)
         
        weight_matrix = self.sign_matrix * efficacy_matrix * self._sigmoid(prob_matrix*self.z_matrix - torch.rand(self.neuron_num, self.neuron_num, device="cpu"), 32)
        
        return weight_matrix


    def _generate_connection_matrix(self):
        """
        Returns a matrix size self.neuron_num by self.neuron_num which tells the type of the connection
        
        1. 0 -> ee
        2. 1 -> ei
        3. 2 -> ie
        4. 3 -> ii
        """
        connection_matrix = torch.zeros((self.neuron_num, self.neuron_num), dtype=torch.int32, device="cpu")

        connection_matrix[self.neuron_num_e:, self.neuron_num_e:] = 3
        connection_matrix[:self.neuron_num_e, self.neuron_num_e:] = 2
        connection_matrix[self.neuron_num_e:, :self.neuron_num_e] = 1
        connection_matrix[:self.neuron_num_e, :self.neuron_num_e] = 0

        return connection_matrix
    

    def _generate_sign_matrix(self):
        """
        Returns a matrix size self.neuron_num by self.neuron_num which tells weight sign
        
        1. 0 -> ee -> +
        2. 1 -> ei -> +
        3. 2 -> ie -> -
        4. 3 -> ii -> -
        """
        sign_matrix = torch.zeros((self.neuron_num, self.neuron_num), dtype=torch.int32, device="cpu")

        sign_matrix[self.neuron_num_e:, self.neuron_num_e:] = -1
        sign_matrix[:self.neuron_num_e, self.neuron_num_e:] = -1
        sign_matrix[self.neuron_num_e:, :self.neuron_num_e] = 1
        sign_matrix[:self.neuron_num_e, :self.neuron_num_e] = 1

        return sign_matrix
    

    def _generate_parameter_matrix(self, params: torch.Tensor):
        """
        Generate matrix of size self.neuron_num by self.neuron_num where 
        each index corresponds to the parameter value which depends on
        the connection type eg. e -> e.
        """
        connection_matrix = self.connection_matrix.type(torch.int64)
        cpu_params = params.to("cpu")
        params_matrix = cpu_params[connection_matrix]
        return params_matrix
    

    def _generate_diff_thetas_matrix(self):
        """
        Generate matrix of size self.neuron_num by self.neuron_num which 
        corresponds to the difference in preffered orientations of the neurons.
        """
        output_orientations = self.pref.repeat(self.neuron_num, 1)
        input_orientations = output_orientations.T
        diff_orientations = torch.abs(input_orientations - output_orientations)
        self.diff_orientations = diff_orientations.to("cpu")
        return self.diff_orientations


    def _generate_z_matrix(self, width: torch.Tensor):
        """
        Generate matrix of size self.neuron_num by self.neuron_num which 
        corresponds to the probability of connection depending on the circular gaussian and
        the difference in preferred orientation.
        """
        self.z_matrix = torch.exp((torch.cos(2 * torch.pi / 180 * self.diff_orientations) - 1) / (4 * (torch.pi / 180 * width)**2))
        return self.z_matrix


    @staticmethod
    def _sigmoid(array, steepness=1):
        """returns the sigmoidal value of the input tensor. The default steepness is 1."""
        return 1 / (1 + torch.exp(-steepness * array))


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
        input_mean = contrast * 20 * self.scaling_g * circ_gauss(grating_orientations - preferred_orientations, self.w_ff)
        input_sd = self.sig_ext
        return input_mean, input_sd


    def _solve_fixed_point(self, input_mean, input_sd): # tau_ref varies with E and I
        r_init = torch.zeros(self.neuron_num, device=self.device) # Need to change this to a matrix
        # Define the function to be solved for
        def drdt_func(rate):
            return self.T_inv * (Phi(*self._get_mu_sigma(self.weights, self.weights2, rate, input_mean, input_sd, self.tau), 
                                     self.tau, 
                                     tau_ref=self.tau_ref,
                                     device=self.device) - rate)
            
        # Solve using Euler
        r_fp, avg_step = Euler2fixedpt(drdt_func, r_init, device=self.device)
        return r_fp, avg_step

    
    # -------------------------RUN OUTPUT TO GET TUNING CURVES--------------------


    def run_all_orientation_and_contrast(self) -> torch.Tensor:
        all_rates = torch.empty(0, device=self.device)
        avg_step_sum = torch.tensor(0, device=self.device)
        count = torch.tensor(0, device=self.device)
        for contrast in self.contrasts:
            steady_states = torch.empty(0, device=self.device)
            for orientation in self.orientations:
                rate, avg_step = self.get_steady_state_output(contrast, orientation)
                steady_states = torch.cat((steady_states, rate.unsqueeze(0)))
                avg_step_sum = avg_step_sum + avg_step
                count = count + torch.tensor(1, device=self.device)
            all_rates = torch.cat((all_rates, steady_states.unsqueeze(0)))
        output = all_rates.permute(2, 0, 1)
        return output, avg_step_sum / count


def training_loop(model, optimizer, Y, n=2000):
    "Training loop for torch model."

    with open(f"log_run_{time.time()}.log", "w") as f:
        loss_function = MMDLossFunction()
        model.train()

        for i in range(n):
            optimizer.zero_grad()
            preds, avg_step = model()
            loss = loss_function(preds, Y, avg_step)
            loss.backward()
            optimizer.step()
            f.write(f"ITTER: {i + 1}  {loss}\n")
            f.write(f"avg step: {avg_step}\n")
            f.write(str(model.j_hyperparameter))
            f.write("\n")
            f.write(str(model.p_hyperparameter))
            f.write("\n")
            f.write(str(model.w_hyperparameter))
            f.write("\n")
            f.write("\n")
            f.flush()
            print(f"DONE {i}")




if __name__ == "__main__":

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

    # J_array = [1.99, 1.9, 1.01, 0.79]  # Need to make parameters of an exponential
    # P_array = [0.11, 0.11, 0.45, 0.45]
    # w_array = [32., 32., 32., 32.]


    # J_array = [0.69, 0.64, 0., -0.29] # Max log values
    # P_array = [-2.21, -2.21, -0.8, -0.8]
    # w_array = [3.46, 3.46, 3.46, 3.46]

    J_array = [0, -0.29, 0.69, -0.64] # Keen log values
    P_array = [-0.8, -2.21, -2.21, -0.8]
    w_array = [3.46, 3.46, 3.46, 3.46]

    # J_array = [-5.1707, -0.0277,  0.9482,  0.1585]
    # P_array = [-6.1361, -1.9772,  0.8548,  1.2384]
    # w_array = [0.3990, 3.6245, 4.1893, 3.6955]

    # J_array = [0.69, 0.64, 0., -0.29]
    # P_array = [-1.21, -1.21, -0.8, -0.8]
    # w_array = [3.46, 3.46, 3.46, 3.46]

    # J_array = [9.04e-02, 3.82e-05, 7.62e-5, 2.52]
    # P_array = [1.93e-02,  8.78,  1.79e-04,  2.85]
    # w_array = [8.78, 166, 1.23e-2, 1.81e2]

    if torch.cuda.is_available():
        device = "cuda"
        print("Model moved to GPU.")
    else:
        device = "cpu"
        print("GPU not available. Keeping the model on CPU.")

    model = NeuroNN(J_array, P_array, w_array, 2000, device=device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    training_loop(model, optimizer, result_array)

    # https://towardsdatascience.com/how-to-use-pytorch-as-a-general-optimizer-a91cbf72a7fb
