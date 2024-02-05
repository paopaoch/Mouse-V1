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

        output.append(torch.tensor(result_array, device=device))

    return output

class MouseLossFunction:
    def __init__(self, avg_step_weighting=0.002, high_contrast_index=7, device="cpu"):
        self.device = device
        self.one = torch.tensor(1)
        self.avg_step_weighting = avg_step_weighting
        self.high_contrast_index = high_contrast_index


    def calculate_loss(self, x_E: torch.Tensor, y_E: torch.Tensor, x_I: torch.Tensor, y_I: torch.Tensor, avg_step: torch.Tensor, x_centralised=False, y_centralised=False):
        if not x_centralised:
            x_E = self.centralise_all_curves(x_E)
            x_I = self.centralise_all_curves(x_I)
        
        if not y_centralised:
            y_E = self.centralise_all_curves(y_E)
            y_I = self.centralise_all_curves(y_I)

        E = self.MMD(x_E, y_E)
        I = self.MMD(x_I, y_I)

        return E + I + (torch.maximum(self.one, avg_step) - 1) * self.avg_step_weighting, E + I

    
    def MMD(self, x: torch.Tensor, y: torch.Tensor):
        XX  = self._individual_terms_single_loop(x, x)
        XY  = self._individual_terms_single_loop(x, y)
        YY  = self._individual_terms_single_loop(y, y)
        return XX + YY - 2 * XY
    

    def _individual_terms_single_loop(self, x: torch.Tensor, y: torch.Tensor):
        N = x.shape[0]
        M = y.shape[0]
        accum_output = torch.tensor(0, device=self.device)
        for i in range(N):
            x_repeated = x[i, :, :].unsqueeze(0).expand(M, -1, -1)
            accum_output = accum_output + torch.mean(self._kernel(y, x_repeated))
        return accum_output / N
    

    @staticmethod
    def _kernel(x, y, w=1, axes=(-2, -1)):
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
    def __init__(self, neuron_num, ratio=0.8, device="cpu"):
        self.neuron_num = neuron_num
        neuron_num_e = int(neuron_num * ratio)
        neuron_num_i = neuron_num - neuron_num_e
        self.neuron_num_e = neuron_num_e
        self.neuron_num_i = neuron_num_i
        self.device = device
        self.ratio = ratio

        self.pref_E = torch.linspace(0, 179.99, self.neuron_num_e, device=device, requires_grad=False)
        self.pref_I = torch.linspace(0, 179.99, self.neuron_num_i, device=device, requires_grad=False)
        self.pref = torch.cat([self.pref_E, self.pref_I]).to(device)


    @staticmethod
    def _cric_gauss(x: torch.Tensor, w):
        """Circular Gaussian from 0 to 180 deg"""
        return torch.exp((torch.cos(x * torch.pi / 90) - 1) / (4 * torch.square(torch.pi / 180 * w)))


    @staticmethod
    def _sigmoid(array: torch.Tensor, steepness=1, scaling=1):
        """returns the sigmoidal value of the input tensor. The default steepness is 1."""
        return scaling / (1 + torch.exp(-steepness * array))

class WeightsGenerator(Rodents):
    def __init__(self, J_array: list, P_array: list, w_array: list, neuron_num: int, ratio=0.8, device="cpu"):
        super().__init__(neuron_num, ratio, device)

        self.j_hyperparameter = torch.tensor(J_array, device=device)
        self.p_hyperparameter = torch.tensor(P_array, device=device)
        self.w_hyperparameter = torch.tensor(w_array, device=device)
    

    def generate_weight_matrix(self):
        prob_EE = self._get_sub_weight_matrix(self._pref_diff(self.pref_E, self.pref_E), 0)
        prob_EI = - self._get_sub_weight_matrix(self._pref_diff(self.pref_E, self.pref_I), 1)
        prob_IE = self._get_sub_weight_matrix(self._pref_diff(self.pref_I, self.pref_E), 2)
        prob_II = - self._get_sub_weight_matrix(self._pref_diff(self.pref_I, self.pref_I), 3)
        weights = torch.cat((torch.cat((prob_EE, prob_EI), dim=1),
                    torch.cat((prob_IE, prob_II), dim=1)), dim=0)

        w_tot_EE = torch.abs(torch.sum(prob_EE)) / self.neuron_num_e
        w_tot_EI = torch.abs(torch.sum(prob_EI)) / self.neuron_num_e
        w_tot_IE = torch.abs(torch.sum(prob_IE)) / self.neuron_num_i
        w_tot_II = torch.abs(torch.sum(prob_IE)) / self.neuron_num_i
        is_valid = ((w_tot_EE / w_tot_IE) < (w_tot_EI / w_tot_II)) < 1


        return weights, is_valid  # Conditions to be changed once we include stimulus
    

    def validate_weights_matrix(self, weights):
        """Return a boolean stating if the matrix is in the valid regime"""
        pass


    def calc_weights_tot(self, weights):
        """Calculate weights tot for the contraints"""
        pass


    def set_parameters(self, J_array, P_array, w_array):
        self.j_hyperparameter = torch.tensor(J_array, device=self.device)
        self.p_hyperparameter = torch.tensor(P_array, device=self.device)
        self.w_hyperparameter = torch.tensor(w_array, device=self.device)
        

    @staticmethod
    def _pref_diff(pref_a: torch.Tensor, pref_b: torch.Tensor) -> torch.Tensor:
        """Create matrix of differences between preferred orientations"""
        return pref_b[None, :] - pref_a[:, None]


    def _get_sub_weight_matrix(self, diff: torch.Tensor, index: int):
        J_single = self._sigmoid(self.j_hyperparameter[index], 1/4, 4) / diff.shape[1]  # could be a sqrt # this is the number of presynaptic neurons
        return J_single * self._sigmoid(self._sigmoid(self.p_hyperparameter[index], 1/3, 1)
                                                                            * self._cric_gauss(diff, self._sigmoid(self.w_hyperparameter[index], 1 / 180, 180))
                                                                            - torch.rand(len(diff), len(diff[0]), device=self.device, requires_grad=False), 32)
    



class NetworkExecuter(Rodents):
    def __init__(self, neuron_num, ratio=0.8, scaling_g=1, w_ff=30, sig_ext=5, device="cpu"):
        super().__init__(neuron_num, ratio, device)

        self.orientations = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
        self.contrasts = [0., 0.0432773, 0.103411, 0.186966, 0.303066, 0.464386, 0.68854, 1.]

        self.weights = None
        self.weights2 = None

        # Stim to inputs
        self.scaling_g = scaling_g * torch.ones(self.neuron_num, device=device, requires_grad=False)
        self.w_ff = w_ff * torch.ones(self.neuron_num, device=device, requires_grad=False)
        self.sig_ext = sig_ext * torch.ones(self.neuron_num, device=device, requires_grad=False)
        
        # Time constants for the ricciardi
        T_alpha = 0.5
        T_E = 0.01
        T_I = 0.01 * T_alpha
        self.T = torch.cat([T_E * torch.ones(self.neuron_num_e, device=device, requires_grad=False), T_I * torch.ones(self.neuron_num_i, device=device, requires_grad=False)])
        self.T_inv = torch.reciprocal(self.T)

        # Membrane time constants for excitatory and inhibitory
        tau_alpha = 1
        tau_E = 0.01
        tau_I = 0.01 * tau_alpha

        # Membrane time constant vector for all cells
        self.tau = torch.cat([tau_E * torch.ones(self.neuron_num_e, device=device, requires_grad=False), tau_I * torch.ones(self.neuron_num_i, device=device, requires_grad=False)])
        self.hardness = 0.01  # So confused
        self.Vt = 20
        self.Vr = 0

        # Refractory periods for exitatory and inhibitory
        tau_ref_E = 0.005
        tau_ref_I = 0.001
        self.tau_ref = torch.cat([tau_ref_E * torch.ones(self.neuron_num_e, device=device, requires_grad=False), tau_ref_I * torch.ones(self.neuron_num_i, device=device, requires_grad=False)])

        # Constants for euler
        self.Nmax=300
        self.Navg=280
        self.dt=0.001
        self.xtol=1e-5
        self.xmin=1e-0

        # Constant for Ricciadi
        self.a = torch.tensor([0.0, 
                    .22757881388024176, .77373949685442023, .32056016125642045, 
                    .32171431660633076, .62718906618071668, .93524391761244940, 
                    1.0616084849547165, .64290613877355551, .14805913578876898], device=device
                    , requires_grad=False)
        

    # -------------------------Public Methods--------------------


    def update_weight_matrix(self, weights) -> None:
        """Update self.weights and self.weights2 which is the weights and weights squares respectively."""
        self.weights = weights
        self.weights2 = torch.square(self.weights)


    def run_all_orientation_and_contrast(self, weights):
        """should condense this down to one single loop like the loss function, 
        runtime will be less but memory might be bad because of (10000, 10000, 12).
        Maybe we can keep the 2 loops but use (10000, 10000, 4) instead. In other words,
        use a third of the orientations at a time then build the matrix up that way.
        """

        if len(weights) != self.neuron_num:
            print(f"ERROR: the object was initialised for {self.neuron_num} neurons but got {len(weights)}")
            return
        else:
            self.update_weight_matrix(weights)
        
        all_rates = []
        avg_step_sum = torch.tensor(0, device=self.device)
        count = 0
        for contrast in self.contrasts:
            steady_states = []
            for orientation in self.orientations:
                rate, avg_step = self._get_steady_state_output(contrast, orientation)
                steady_states.append(rate)
                avg_step_sum = avg_step_sum + avg_step
                count += 1
            all_rates.append(torch.stack(steady_states))
        output  = torch.stack(all_rates).permute(2, 0, 1)
        print("finished all orientations and contrasts", output.shape)
        return output, avg_step_sum / count


    #---------------------RUN THE NETWORK TO GET STEADY STATE OUTPUT------------------------


    def _get_steady_state_output(self, contrast, grating_orientations):
        self._stim_to_inputs(contrast, grating_orientations, self.pref)
        r_fp, avg_step = self._solve_fixed_point()
        return r_fp, avg_step


    def _update_mu_sigma(self, rate):
        # Find net input mean and variance given inputs
        self.mu = self.tau * (self.weights @ rate) + self.input_mean
        self.sigma = torch.sqrt(self.tau * (self.weights2 @ rate) + torch.square(self.input_sd))
        return self.mu, self.sigma
    

    def _stim_to_inputs(self, contrast, grating_orientations, preferred_orientations):
        '''Set the inputs based on the contrast and orientation of the stimulus'''
        self.input_mean = contrast * 20 * self.scaling_g * self._cric_gauss(grating_orientations - preferred_orientations, self.w_ff)
        self.input_sd = self.sig_ext
        return self.input_mean, self.input_sd


    def _solve_fixed_point(self): # tau_ref varies with E and I
        r_init = torch.zeros(self.neuron_num, device=self.device) # Need to change this to a matrix
        # Solve using Euler
        r_fp, avg_step = self._euler2fixedpt(r_init)
        return r_fp, avg_step
    

    # -------------------------MOVE SIM UTILS INTO THE SAME CLASS------------------

    
    @staticmethod
    def _softplus(x, b):
        return torch.log(1 + torch.exp(b * x)) / b
    

    def _euler2fixedpt(self, x_initial):
        xmin = torch.tensor(self.xmin, device=self.device, requires_grad=False)

        avgStart = self.Nmax - self.Navg
        avg_sum = 0
        xvec = x_initial

        for _ in range(avgStart):  # Loop without taking average step size
            self._update_mu_sigma(xvec)
            dx = self.T_inv * (self._phi() - xvec) * self.dt
            xvec = xvec + dx

        for _ in range(self.Navg):  # Loop whilst recording average step size
            self._update_mu_sigma(xvec)
            dx = self.T_inv * (self._phi() - xvec) * self.dt
            xvec = xvec + dx
            avg_sum = avg_sum + torch.abs(dx / torch.maximum(xmin, torch.abs(xvec)) ).max() / self.xtol

        return xvec, avg_sum / self.Navg


    # This is the input-output function (for mean-field spiking neurons) that you would use Max
    def _phi(self):

        # Might need error handling for mu and sigma being None
        xp = (self.mu - self.Vr) / self.sigma
        xm = (self.mu - self.Vt) / self.sigma
        

        # rate = torch.zeros_like(xm, device=self.device) # dunno why we need this?
        xm_pos = self._sigmoid(xm * self.hardness)
        inds = self._sigmoid(-xm * self.hardness) * self._sigmoid(xp * self.hardness)
        
        xp1 = self._softplus(xp, self.hardness)
        xm1 = self._softplus(xm, self.hardness)
        
        #xm_pos = xm > 0
        # rate = (rate * (1 - xm_pos)) + (xm_pos / self._softplus(self._f_ricci(xp1) - self._f_ricci(xm1), self.hardness))
        rate = (xm_pos / self._softplus(self._f_ricci(xp1) - self._f_ricci(xm1), self.hardness))


        #inds = (xp > 0) & (xm <= 0)
        rate = (rate * (1 - inds)) + (inds / (self._f_ricci(xp1) + torch.exp(xm**2) * self._g_ricci(self._softplus(-xm, self.hardness))))
        
        rate = 1 / (self.tau_ref + 1 / rate)

        return rate / self.tau


    def _f_ricci(self, x):
        z = x / (1 + x)
        return torch.log(2*x + 1) + (self.a[1] *(-z)**1 + self.a[2] *(-z)**2 + self.a[3] *(-z)**3
                                + self.a[4] *(-z)**4 + self.a[5] *(-z)**5 + self.a[6] *(-z)**6
                                + self.a[7] *(-z)**7 + self.a[8] *(-z)**8 + self.a[9] *(-z)**9)

    @staticmethod
    def _g_ricci(x):

        z = x / (2 + x)
        enum = (  3.5441754117462949 * z    - 7.0529131065835378 * z**2 
                - 56.532378057580381 * z**3 + 279.56761105465944 * z**4 
                - 520.37554849441472 * z**5 + 456.58245777026514 * z**6  
                - 155.73340457809226 * z**7 )
        
        denom = (1 - 4.1357968834226053 * z - 7.2984226138266743 * z**2 
                + 98.656602235468327 * z**3 - 334.20436223415163 * z**4 
                + 601.08633903294185 * z**5 - 599.58577549598340 * z**6 
                + 277.18420330693891 * z**7 - 16.445022798669722 * z**8)
        
        return enum / denom




if __name__ == "__main__":
    J_array = [-5.865,-7.78, 0, -4.39]
    P_array = [0, 0, 0, 0]
    w_array = [-45.94, -60, -35.69, -45.94]
    # J_array = [  0.6131,  -6.8548,   2.2939,  -5.6821]  # NES values  # These values are wrong
    # P_array = [-0.6996,  -7.7089,  -1.3388, -4.4278] 
    # w_array = [-12.3577, -15.2088,  -9.6759, -14.3590]

    keen = WeightsGenerator(J_array, P_array, w_array, 10000)
    W, accepted = keen.generate_weight_matrix()

    print(accepted)

    # executer = NetworkExecuter(W)
    # responses, avg_step = executer.run_all_orientation_and_contrast()
    # print(responses.shape, avg_step)

    # one_res = []
    # for i in range(10000):
    #     one_res.append(responses[i][7][4])
    
    # plt.plot(one_res)
    # plt.title("Activity of the network")
    # plt.xlabel("Neuron Index")
    # plt.ylabel("Response / Hz")
    # plt.show()

    plt.imshow(W, cmap="seismic", vmin=-np.max(np.abs(np.array(W))), vmax=np.max(np.abs(np.array(W))))
    plt.colorbar()
    plt.title(f"Connection weight matrix for {10000} neurons")
    plt.xlabel("Neuron index")
    plt.ylabel("Neuron index")
    plt.show()