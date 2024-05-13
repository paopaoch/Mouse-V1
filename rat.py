"""
RAT

This file contains all the necessary class and functions for the newer version of the mouse V1 project. This includes:

get_data: function which returns tuning curves from the data

MouseLossFunction: The loss function which incorporates the MMD loss for E and I neuron separately and the average step from running
                    euler to fixed point

Rodents: Parent class containing the hyperparameters and initialisation.

ConnectivityWeights: Inherit from Rodents for weight matrix generation.

NetworkExecuter: Inherit from Rodents for running the network given a weight matrix.

NOTE: THIS FILE DOES NOT INHERIT FROM torch.nn AND DUE TO RANDOMNESS, SOME FUNCTIONS DO NOT SUPPORT BACKPROPAGATION AND GRADIENT BASED OPTIMISATION.

"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep

# torch.set_default_dtype(torch.float32)

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


class MouseLossFunction:
    def __init__(self, avg_step_weighting=0.002, high_contrast_index=7, device="cpu"):
        self.device = device
        self.one = torch.tensor(1)
        self.avg_step_weighting = avg_step_weighting
        self.high_contrast_index = high_contrast_index


    def calculate_loss(self, x_E: torch.Tensor, y_E: torch.Tensor, x_I: torch.Tensor, y_I: torch.Tensor, avg_step: torch.Tensor, bessel_val=torch.tensor(0), bessel_val_weighting=torch.tensor(1), x_centralised=False, y_centralised=False):
        if not x_centralised:
            x_E = self.centralise_all_curves(x_E)
            x_I = self.centralise_all_curves(x_I)
        
        if not y_centralised:
            y_E = self.centralise_all_curves(y_E)
            y_I = self.centralise_all_curves(y_I)

        E = self.MMD(x_E, y_E)
        I = self.MMD(x_I, y_I)

        return E + I + (torch.maximum(self.one, avg_step) - 1) * self.avg_step_weighting + bessel_val * bessel_val_weighting, E + I

    
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


class MouseLossFunctionOptimised(MouseLossFunction):


    def MMD(self, x: torch.Tensor, y: torch.Tensor):
        x = self.reshape_for_optimised(x)  # TODO: When we use this MMD implementation fully, reshape this outside
        y = self.reshape_for_optimised(y)

        G_x = x @ x.T
        G_y = y @ y.T
        G_xy = x @ y.T
        return torch.mean(self._optimised_kernel(G_x, G_x, G_x)) + torch.mean(self._optimised_kernel(G_y, G_y, G_y)) - torch.mean(2*self._optimised_kernel(G_x, G_y, G_xy))


    @staticmethod
    def _optimised_kernel(U, V, W, sigma=0.5):
        return torch.exp(-sigma * (torch.diag(U)[:, None]
                               + torch.diag(V)[None, :]
                               - 2* W))

    
    @staticmethod
    def reshape_for_optimised(x: torch.Tensor):
        batch_size, dim1, dim2 = x.size()
        return torch.reshape(x, (batch_size, dim1 * dim2))
    

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


class WeightsGenerator(Rodents):
    def __init__(self, J_array: list, P_array: list, w_array: list, neuron_num: int, feed_forward_num=100, ratio=0.8, device="cpu", requires_grad=False, forward_mode=False):
        super().__init__(neuron_num, ratio, device, feed_forward_num)

        if len(J_array) != len(P_array) or len(P_array) != len(w_array):
            raise IndexError("Expect all parameter arrays to be the same length.")
        if len(J_array) != 4 and len(J_array) != 6:
            raise IndexError(f"Expect length of parameter arrays to be 4 or 6 but received {len(J_array)}.")

        if forward_mode:
            self.J_parameters = J_array
            self.P_parameters = P_array
            self.w_parameters = w_array
        else:
            self.J_parameters = torch.tensor(J_array, device=device, requires_grad=requires_grad)
            self.P_parameters = torch.tensor(P_array, device=device, requires_grad=requires_grad)
            self.w_parameters = torch.tensor(w_array, device=device, requires_grad=requires_grad)

        # Sigmoid values for parameters
        self.J_steep = 1
        self.J_scale = 100

        self.P_steep = 1
        self.P_scale = 1

        self.w_steep = 1
        self.w_scale = 180


    def generate_weight_matrix(self):
        prob_EE = self._get_sub_weight_matrix(self._pref_diff(self.pref_E, self.pref_E), 0)
        prob_EI = - self._get_sub_weight_matrix(self._pref_diff(self.pref_E, self.pref_I), 1)
        prob_IE = self._get_sub_weight_matrix(self._pref_diff(self.pref_I, self.pref_E), 2)
        prob_II = - self._get_sub_weight_matrix(self._pref_diff(self.pref_I, self.pref_I), 3)
        weights = torch.cat((torch.cat((prob_EE, prob_EI), dim=1),
                    torch.cat((prob_IE, prob_II), dim=1)), dim=0)
        return weights


    def validate_weight_matrix(self):
        W_tot_EE = self.calc_theoretical_weights_tot(0, self.neuron_num_e)
        W_tot_EI = self.calc_theoretical_weights_tot(1, self.neuron_num_i)
        W_tot_IE = self.calc_theoretical_weights_tot(2, self.neuron_num_e)
        W_tot_II = self.calc_theoretical_weights_tot(3, self.neuron_num_i)

        if len(self.J_parameters) == 6:
            W_tot_EF = self.calc_theoretical_weights_tot(4, self.feed_forward_num)
            W_tot_IF = self.calc_theoretical_weights_tot(5, self.feed_forward_num)
        else:
            W_tot_EF = torch.tensor(1, device=self.device)
            W_tot_IF = torch.tensor(1, device=self.device)
        
        first_condition = torch.maximum((W_tot_EE / W_tot_IE) - (W_tot_EI / W_tot_II), torch.tensor(0, device=self.device))
        second_condition = torch.maximum((W_tot_EI / W_tot_II) - (W_tot_EF / W_tot_IF), torch.tensor(0, device=self.device))
        return torch.maximum(first_condition, second_condition)


    def balance_in_ex_in(self):
        unscaled_W_tot_EE = self.calc_theoretical_weights_tot(0, self.neuron_num_e, scale=False)
        unscaled_W_tot_EI = self.calc_theoretical_weights_tot(1, self.neuron_num_i, scale=False)
        unscaled_W_tot_IE = self.calc_theoretical_weights_tot(2, self.neuron_num_e, scale=False)
        unscaled_W_tot_II = self.calc_theoretical_weights_tot(3, self.neuron_num_i, scale=False)
        
        return unscaled_W_tot_EE, unscaled_W_tot_EI, unscaled_W_tot_IE, unscaled_W_tot_II
    

    def calc_theoretical_weights_tot(self, i, N_b, scale=True):
        """Calculate weights tot for the contraints"""
        k = 1 / (4 * (self._sigmoid(self.w_parameters[i], self.w_steep, self.w_scale) * torch.pi / 180) ** 2)
        
        bessel: torch.Tensor = torch.special.i0(k)

        j = self._sigmoid(self.J_parameters[i], self.J_steep, self.J_scale)
        p = self._sigmoid(self.P_parameters[i], self.P_steep, self.P_scale)
        if scale:
            return j * torch.sqrt(torch.tensor(N_b, device=self.device)) * p * torch.exp(-k) * bessel
        else:
            return torch.sqrt(torch.tensor(N_b, device=self.device)) * p * torch.exp(-k) * bessel


    def set_parameters(self, J_array, P_array, w_array):
        self.J_parameters = torch.tensor(J_array, device=self.device)
        self.P_parameters = torch.tensor(P_array, device=self.device)
        self.w_parameters = torch.tensor(w_array, device=self.device)
        

    @staticmethod
    def _pref_diff(pref_a: torch.Tensor, pref_b: torch.Tensor) -> torch.Tensor:
        """Create matrix of differences between preferred orientations"""
        return pref_b[None, :] - pref_a[:, None]


    def _get_sub_weight_matrix(self, diff: torch.Tensor, index: int):  # Rewrite this lmao
        J_single = self._sigmoid(self.J_parameters[index], self.J_steep, self.J_scale) / torch.sqrt(torch.tensor(diff.shape[1]))  # dont have to be a sqrt # this is the number of presynaptic neurons
        return J_single * self._sigmoid(self._sigmoid(self.P_parameters[index], self.P_steep, self.P_scale)
                                        * self._cric_gauss(diff, self._sigmoid(self.w_parameters[index], self.w_steep, self.w_scale))
                                        - torch.rand(len(diff), len(diff[0]), device=self.device, requires_grad=False), 32)


class WeightsGeneratorExact(WeightsGenerator):
    def _get_sub_weight_matrix(self, diff: torch.Tensor, index: int):
        J_single = self._sigmoid(self.J_parameters[index], self.J_steep, self.J_scale) / torch.sqrt(torch.tensor(diff.shape[1]))
        return J_single * torch.bernoulli(self._sigmoid(self.P_parameters[index], self.P_steep, self.P_scale) 
                                          * self._cric_gauss(diff, self._sigmoid(self.w_parameters[index], self.w_steep, self.w_scale)))


class RandomWeightsGenerator(WeightsGenerator):
    def _get_sub_weight_matrix(self, diff: torch.Tensor, index: int):
        J_single = self._sigmoid(self.J_parameters[index], self.J_steep, self.J_scale) / torch.sqrt(torch.tensor(diff.shape[1]))
        prob_matrix = torch.ones_like(diff) * self._sigmoid(self.P_parameters[index], self.P_steep, self.P_scale)
        return J_single * torch.bernoulli(prob_matrix)


class NetworkExecuter(Rodents):
    def __init__(self, neuron_num, feed_forward_num=100, ratio=0.8, scaling_g=0.15, w_ff=30, sig_ext=5, device="cpu", plot_overtime=False):
        super().__init__(neuron_num, ratio, device, feed_forward_num)

        self.orientations = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]  # 12  # NOTE: we can reduce this for experimental runs, we can change this as we move closer to the optimal during optimisation
        self.contrasts = [0, 0.0432773, 0.103411, 0.186966, 0.303066, 0.464386, 0.68854, 1.]  # 8
        # self.contrasts = [0., 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1.]  # 8
        # self.contrasts = [.3, .4, .5, .6, .7, .8, .9, 1.]  # 8

        self.weights = None
        self.weights2 = None
        self.weights_FF = None
        self.weights_FF2 = None

        # Stim to inputs
        self.scaling_g = torch.tensor(scaling_g, device=device)
        self.w_ff = torch.tensor(w_ff, device=device)
        self.sig_ext = torch.tensor(sig_ext, device=device)
        
        # Time constants for the ricciardi
        T_alpha = 0.5  # NOTE: Change this to change convergence (reduce this could make faster convergence)
        T_E = 0.01
        T_I = 0.01 * T_alpha
        self.T = torch.cat([T_E * torch.ones(self.neuron_num_e, device=device, requires_grad=False), T_I * torch.ones(self.neuron_num_i, device=device, requires_grad=False)])
        self.T_inv = torch.reciprocal(self.T)

        # Membrane time constants for excitatory and inhibitory
        tau_alpha = 1
        tau_E = 0.01
        tau_I = 0.01 * tau_alpha

        # Membrane time constant vector for all cells
        self.tau = torch.cat([tau_E * torch.ones(self.neuron_num_e, device=device, requires_grad=False),
                              tau_I * torch.ones(self.neuron_num_i, device=device, requires_grad=False)])
        # self.hardness = 0.01  # So confused
        self.hardness = 50
        self.Vt = 20
        self.Vr = 0

        # Refractory periods for exitatory and inhibitory
        tau_ref_E = 0.005
        tau_ref_I = 0.001
        self.tau_ref = torch.cat([tau_ref_E * torch.ones(self.neuron_num_e, device=device, requires_grad=False), 
                                  tau_ref_I * torch.ones(self.neuron_num_i, device=device, requires_grad=False)])

        # Constants for euler
        self.Nmax=300
        self.Navg=280
        self.dt=0.001
        self.xmin=1e-0

        # Constant for Ricciadi
        self.a = torch.tensor([0.0, 
                    .22757881388024176, .77373949685442023, .32056016125642045, 
                    .32171431660633076, .62718906618071668, .93524391761244940, 
                    1.0616084849547165, .64290613877355551, .14805913578876898], device=device
                    , requires_grad=False)
        
        # Add noise to fix point output
        self.N_trial = 50
        self.recorded_spike_T = 0.5
        self.plot_overtime = plot_overtime
        

    # -------------------------Public Methods--------------------


    def update_weight_matrix(self, weights, weights_FF=None) -> None:
        """Update self.weights and self.weights2 which is the weights and weights squares respectively."""
        self.weights = weights
        self.weights2 = torch.square(self.weights)
        if weights_FF is not None:
            self.weights_FF = weights_FF
            self.weights_FF2 = torch.square(weights_FF)
        else:
            self.weights_FF = None
            self.weights_FF2 = None


    def run_all_orientation_and_contrast(self, weights, weights_FF=None):
        """should condense this down to one single loop like the loss function, 
        runtime will be less but memory might be bad because of (10000, 10000, 12).
        Maybe we can keep the 2 loops but use (10000, 10000, 4) instead. In other words,
        use a third of the orientations at a time then build the matrix up that way.
        """

        if len(weights) != self.neuron_num:
            print(f"ERROR: the object was initialised for {self.neuron_num} neurons but got {len(weights)}")
            return
        else:
            self.update_weight_matrix(weights, weights_FF)
        
        all_rates = []
        avg_step_sum = torch.tensor(0, device=self.device)
        count = 0
        for contrast in self.contrasts:
            steady_states = []
            for orientation in self.orientations:
                rate, avg_step = self._get_steady_state_output(contrast, orientation)
                rate = self._add_noise_to_rate(rate)
                steady_states.append(rate)
                avg_step_sum = avg_step_sum + avg_step
                count += 1
            all_rates.append(torch.stack(steady_states))
        output  = torch.stack(all_rates).permute(2, 0, 1)
        return output, avg_step_sum / count


    #---------------------RUN THE NETWORK TO GET STEADY STATE OUTPUT------------------------


    def _get_steady_state_output(self, contrast, grating_orientations):
        if self.weights_FF is None:
            self._stim_to_inputs(contrast, grating_orientations)
        else:
            self._stim_to_inputs_with_ff(contrast, grating_orientations)
        r_fp, avg_step = self._solve_fixed_point()
        return r_fp, avg_step


    def _update_mu_sigma(self, rate):
        # Find net input mean and variance given inputs
        self.mu = self.tau * (self.weights @ rate) + self.input_mean
        self.sigma = torch.sqrt(self.tau * (self.weights2 @ rate) + torch.square(self.input_sd))
        return self.mu, self.sigma
    

    def _stim_to_inputs(self, contrast, grating_orientation):
        '''Set the inputs based on the contrast and orientation of the stimulus'''
        # mu_BL = 4.16
        mu_BL = self.Vt - 1.5 * self.sig_ext  # NOTE: try 2.71, 2, 1.5
        self.input_mean = mu_BL + contrast * (self.Vt - self.Vr) * self.scaling_g * self._cric_gauss(grating_orientation - self.pref, self.w_ff)
        self.input_sd = self.sig_ext
        return self.input_mean, self.input_sd
    

    def _stim_to_inputs_with_ff(self, contrast, grating_orientation):
        '''Set the inputs based on the contrast and orientation of the stimulus'''
        ff_output = (self.Vt - self.Vr) * contrast * self.scaling_g * self._cric_gauss(grating_orientation - self.pref_F, self.w_ff)
        self.input_mean = self.weights_FF @ ff_output
        self.input_sd = self.weights_FF2 @ ff_output + torch.tensor(0.01, device=self.device)   # Adding a small DC offset here to prevent division by 0 error

        return self.input_mean, self.input_sd


    def _solve_fixed_point(self):
        r_init = torch.zeros(self.neuron_num, device=self.device)
        r_fp, avg_step = self._euler2fixedpt(r_init)
        return r_fp, avg_step
    
    
    def _add_noise_to_rate(self, rate_fp: torch.Tensor):
        sigma = torch.sqrt(rate_fp / self.N_trial / self.recorded_spike_T)
        rand = torch.randn(size=rate_fp.shape, device=self.device)
        return torch.abs(rate_fp + sigma * rand)  # TODO: check this inclusion of the abs
    

    # -------------------------MOVE SIM UTILS INTO THE SAME CLASS------------------

    
    @staticmethod
    def _softplus(x, b):
        return torch.log(1 + torch.exp(b * x)) / b
    

    def _euler2fixedpt(self, xvec):
        xmin = torch.tensor(self.xmin, device=self.device, requires_grad=False)

        avgStart = self.Nmax - self.Navg
        avg_sum = 0

        recorded_xvec = []

        for _ in range(avgStart):  # Loop without taking average step size
            self._update_mu_sigma(xvec)
            dx = self.T_inv * (self._phi() - xvec) * self.dt
            xvec = xvec + dx
            recorded_xvec.append(xvec)

        for _ in range(self.Navg):  # Loop whilst recording average step size
            self._update_mu_sigma(xvec)
            dx = self.T_inv * (self._phi() - xvec) * self.dt
            xvec = xvec + dx
            avg_sum = avg_sum + torch.abs(dx / torch.maximum(xmin, torch.abs(xvec)) ).max()
            recorded_xvec.append(xvec)

        if self.plot_overtime:
            return recorded_xvec, 0

        return xvec, avg_sum / self.Navg
    
    
    @staticmethod
    def _relu(x):
        return torch.max(torch.tensor(0.0), x)
    
    @staticmethod
    def _step(x):
        return torch.where(x >= 0, torch.tensor(1.0), torch.tensor(0.0))


    def _phi(self):
        xp = (self.mu - self.Vr) / self.sigma
        xm = (self.mu - self.Vt) / self.sigma

        rate = torch.zeros_like(xm)
        rate[xm > 0] = 1 / (self._f_ricci(xp[xm > 0]) - self._f_ricci(xm[xm > 0]))
        inds = (xp > 0) & (xm <= 0)
        rate[inds] = 1 / ( self._f_ricci(xp[inds]) + torch.exp(xm[inds]**2) * self._g_ricci(-xm[inds]) )
        rate[xp <= 0] = torch.exp(-xm[xp <= 0]**2 - torch.log(self._g_ricci(-xm[xp <= 0]) 
                            - torch.exp(xp[xp <= 0]**2 - xm[xp <= 0]**2) * self._g_ricci(-xp[xp <= 0])))
        
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
    

class NetworkExecuterParallel(NetworkExecuter):

    def __init__(self, neuron_num, feed_forward_num=100, ratio=0.8, scaling_g=0.15, w_ff=30, sig_ext=5, device="cpu", plot_overtime=False):
        super().__init__(neuron_num, feed_forward_num, ratio, scaling_g, w_ff, sig_ext, device, plot_overtime)
        self.tau = self.tau.unsqueeze(0)
        self.tau = self.tau.repeat(len(self.orientations) * len(self.contrasts), 1).T

        self.tau_ref = self.tau_ref.unsqueeze(0)
        self.tau_ref = self.tau_ref.repeat(len(self.orientations) * len(self.contrasts), 1).T

        self.T_inv = self.T_inv.unsqueeze(0)
        self.T_inv = self.T_inv.repeat(len(self.orientations) * len(self.contrasts), 1).T

    def run_all_orientation_and_contrast(self, weights, weights_FF=None):
        if len(weights) != self.neuron_num:
            print(f"ERROR: the object was initialised for {self.neuron_num} neurons but got {len(weights)}")
            return
        else:
            self.update_weight_matrix(weights, weights_FF)

        rate, avg_step = self._get_steady_state_output()
        rate = rate.view(self.neuron_num, 8, 12)
        return self._add_noise_to_rate(rate), avg_step
    

    def _stim_to_inputs(self):
        '''Set the inputs based on the contrast and orientation of the stimulus'''
        input_mean = []
        # mu_BL = 4.16
        mu_BL = self.Vt - 2.71 * self.sig_ext
        for contrast in self.contrasts:
            for orientation in self.orientations:
                input_mean.append(mu_BL * contrast * (self.Vt - self.Vr) * self.scaling_g * self._cric_gauss(orientation - self.pref, self.w_ff))  # NOTE: replace 20 with v_t - v_r
        self.input_mean = torch.stack(input_mean).T  # Dont forget to transpose back to get input for each constrast and orientation
        self.input_sd = self.sig_ext  # THIS DEFAULTS TO 5
        return self.input_mean, self.input_sd


    def _stim_to_inputs_with_ff(self):
        '''Set the inputs based on the contrast and orientation of the stimulus'''
        input_mean = []
        mu_BL = 1
        for contrast in self.contrasts:
            for orientation in self.orientations:
                input_mean.append((self.Vt - self.Vr) * mu_BL * contrast * self.scaling_g * self._cric_gauss(orientation - self.pref_F, self.w_ff))  # NOTE: Check this
        ff_output = torch.stack(input_mean).T
        self.input_mean = self.weights_FF @ ff_output
        self.input_sd = self.weights_FF2 @ ff_output + torch.tensor(0.01, device=self.device)  # Adding a small DC offset here to prevent division by 0 error

        return self.input_mean, self.input_sd
    

    def _get_steady_state_output(self):
        if self.weights_FF is None:
            self._stim_to_inputs()
        else:
            self._stim_to_inputs_with_ff()
        r_fp, avg_step = self._solve_fixed_point()
        return r_fp, avg_step


    def _solve_fixed_point(self):
        r_init = torch.zeros_like(self.input_mean, device=self.device)
        r_fp, avg_step = self._euler2fixedpt(r_init)
        return r_fp, avg_step
    

    def _update_mu_sigma(self, rate):
        # Find net input mean and variance given inputs
        self.mu = self.tau * (self.weights @ rate) + self.input_mean  # TODO: Check why tau is there mathematically
        self.sigma = torch.sqrt(self.tau * (self.weights2 @ rate) + torch.square(self.input_sd))
        return self.mu, self.sigma


if __name__ == "__main__":
    from rodents_plotter import plot_weights, print_tuning_curve, centralise_all_curves, print_activity
    from time import time

    # J_array = [-4.054651081081644, -19.924301646902062, -0.0, -12.083112059245341]
    # P_array = [-6.591673732008658, 1.8571176252186712, -4.1588830833596715, 4.549042468104266]
    # w_array = [-167.03761889472233, -187.23627477210516, -143.08737747657977, -167.03761889472233]

    J_array = [0.5, -0.5, -0.5, 0.5]
    P_array = [-0.5, -0.5, -0.5, -0.5]
    w_array = [-275.66574677358994, -275.66574677358994, -275.66574677358994, -275.66574677358994]

    n = 100

    keen = RandomWeightsGenerator(J_array, P_array, w_array, n, 100, device="cpu")
    W = keen.generate_weight_matrix()
    plot_weights(W, title="")