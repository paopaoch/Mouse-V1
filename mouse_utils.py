import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from tqdm import tqdm
import sys

# Set the default data type to float32 globally
torch.set_default_dtype(torch.float32)

class MMDLossFunction(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.one = torch.tensor(1)

    
    def forward(self, x: torch.Tensor, y: torch.Tensor, avg_step: torch.Tensor):
        XX  = self.individual_terms_single_loop(x, x)
        XY  = self.individual_terms_single_loop(x, y)
        YY  = self.individual_terms_single_loop(y, y)
        return XX + YY - 2 * XY  + (torch.maximum(self.one, avg_step) - 1) * 0.002
    

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


class NeuroNN(nn.Module):
    """
    ### This class wraps the logic for the forward pass for modelling the mouse V1

    The forward pass performs two computations:
    1. Update the weight matrix
    2. Solves for fixed point at all contrast and orientation combinations
    """

    def __init__(self, J_array: list, P_array: list, w_array: list, neuron_num: int, ratio=0.8, scaling_g=1, w_ff=30, sig_ext=5, device="cpu", grad=True):
        super().__init__()
        self.device = device
        self.orientations = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
        self.contrasts = [0., 0.0432773, 0.103411, 0.186966, 0.303066, 0.464386, 0.68854, 1.]

        j_hyper = torch.tensor(J_array, device="cpu")  # offload GPU by having W matrix generation on CPU
        p_hyper = torch.tensor(P_array, device="cpu")
        w_hyper = torch.tensor(w_array, device="cpu")
        if grad:
            self.j_hyperparameter = nn.Parameter(j_hyper)
            self.p_hyperparameter = nn.Parameter(p_hyper)
            self.w_hyperparameter = nn.Parameter(w_hyper)
        else:  # set grad to False in-case we dont want to do backwards
            self.j_hyperparameter = j_hyper
            self.p_hyperparameter = p_hyper
            self.w_hyperparameter = w_hyper


        self.neuron_num = neuron_num
        neuron_num_e = int(neuron_num * ratio)
        neuron_num_i = neuron_num - neuron_num_e
        self.neuron_num_e = neuron_num_e
        self.neuron_num_i = neuron_num_i

        self.pref_E = torch.linspace(0, 179.99, neuron_num_e, device="cpu", requires_grad=False)
        self.pref_I = torch.linspace(0, 179.99, neuron_num_i, device="cpu", requires_grad=False)
        self.pref = torch.cat([self.pref_E, self.pref_I]).to(device)

        # Global Parameters
        self.scaling_g = scaling_g * torch.ones(neuron_num, device=device, requires_grad=False)
        self.w_ff = w_ff * torch.ones(neuron_num, device=device, requires_grad=False)
        self.sig_ext = sig_ext * torch.ones(neuron_num, device=device, requires_grad=False)
        
        T_alpha = 0.5
        T_E = 0.01
        T_I = 0.01 * T_alpha
        self.T = torch.cat([T_E * torch.ones(neuron_num_e, device=device, requires_grad=False), T_I * torch.ones(neuron_num_i, device=device, requires_grad=False)])
        self.T_inv = torch.reciprocal(self.T)

        # Membrane time constants for excitatory and inhibitory
        tau_alpha = 1
        tau_E = 0.01
        tau_I = 0.01 * tau_alpha
        # Membrane time constant vector for all cells
        self.tau = torch.cat([tau_E * torch.ones(neuron_num_e, device=device, requires_grad=False), tau_I * torch.ones(neuron_num_i, device=device, requires_grad=False)])
        self.hardness = 0.01  # So confused
        self.Vt = 20
        self.Vr = 0

        # Refractory periods for exitatory and inhibitory
        tau_ref_E = 0.005
        tau_ref_I = 0.001
        self.tau_ref = torch.cat([tau_ref_E * torch.ones(neuron_num_e, device=device, requires_grad=False), tau_ref_I * torch.ones(neuron_num_i, device=device, requires_grad=False)])

        self.weights = None
        self.weights2 = None
        self.update_weight_matrix()

        # Constant for Ricciadi
        self.a = torch.tensor([0.0, 
                    .22757881388024176, .77373949685442023, .32056016125642045, 
                    .32171431660633076, .62718906618071668, .93524391761244940, 
                    1.0616084849547165, .64290613877355551, .14805913578876898], device=device
                    , requires_grad=False)

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
        print("Generated a W matrix")
        tuning_curves, avg_step = self.run_all_orientation_and_contrast()
        return tuning_curves, avg_step


    # ------------------GET WEIGHT MATRIX--------------------------


    def update_weight_matrix(self) -> None:
        """Update self.weights and self.weights2 which is the weights and weights squares respectively."""
        self.weights = self.generate_weight_matrix().to(self.device)
        self.weights2 = torch.square(self.weights)


    @staticmethod
    def pref_diff(pref_a: torch.Tensor, pref_b: torch.Tensor) -> torch.Tensor:
        """Create matrix of differences between preferred orientations"""
        return pref_b[None, :] - pref_a[:, None]


    def _get_sub_weight_matrix(self, diff: torch.Tensor, index: int):
        return torch.exp(self.j_hyperparameter[index]) * self._sigmoid(self._sigmoid(self.p_hyperparameter[index], 2)
                                                                       * self._cric_gauss(diff, torch.exp(self.w_hyperparameter[index])) 
                                                            - torch.rand(len(diff), len(diff[0]), device="cpu", requires_grad=False), 32)


    def generate_weight_matrix(self):
        prob_EE = self._get_sub_weight_matrix(self.pref_diff(self.pref_E, self.pref_E), 0)
        prob_EI = self._get_sub_weight_matrix(self.pref_diff(self.pref_E, self.pref_I), 1)
        prob_IE = - self._get_sub_weight_matrix(self.pref_diff(self.pref_I, self.pref_E), 2)
        prob_II = - self._get_sub_weight_matrix(self.pref_diff(self.pref_I, self.pref_I), 3)
        weights = torch.transpose(torch.cat((torch.cat((prob_EE, prob_EI), dim=1),
                    torch.cat((prob_IE, prob_II), dim=1)), dim=0), 0, 1)
        
        return weights


    @staticmethod
    def _cric_gauss(x: torch.Tensor, w):
        # Circular Gaussian from 0 to 180 deg
        return torch.exp((torch.cos(x * torch.pi / 90) - 1) / (4 * torch.square(torch.pi / 180 * w)))


    @staticmethod
    def _sigmoid(array: torch.Tensor, steepness=1):
        """returns the sigmoidal value of the input tensor. The default steepness is 1."""
        return 1 / (1 + torch.exp(-steepness * array))


    #---------------------RUN THE NETWORK TO GET STEADY STATE OUTPUT------------------------


    def get_steady_state_output(self, contrast, grating_orientations):
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


    def drdt_func(self, rate):
        self._update_mu_sigma(rate)
        return self.T_inv * (self.phi() - rate)


    def _solve_fixed_point(self): # tau_ref varies with E and I
        r_init = torch.zeros(self.neuron_num, device=self.device) # Need to change this to a matrix
        # Solve using Euler
        r_fp, avg_step = self.euler2fixedpt(r_init)
        return r_fp, avg_step

    
    # -------------------------RUN OUTPUT TO GET TUNING CURVES--------------------


    def run_all_orientation_and_contrast(self):
        all_rates = []
        avg_step_sum = torch.tensor(0, device=self.device)
        count = 0
        for contrast in self.contrasts:
            steady_states = []
            for orientation in self.orientations:
                rate, avg_step = self.get_steady_state_output(contrast, orientation)
                steady_states.append(rate)
                avg_step_sum = avg_step_sum + avg_step
                count += 1
            all_rates.append(torch.stack(steady_states))
        output  = torch.stack(all_rates).permute(2, 0, 1)
        print("finished all orientations and contrasts", output.shape)
        return output, avg_step_sum / count
    

    # -------------------------MOVE SIM UTILS INTO THE SAME CLASS------------------

    @staticmethod
    def sigmoid(x):  # similar to _sigmoid so perhaps combine in the future
        return 1 / (1 + torch.exp(-x))

    
    @staticmethod
    def softplus(x, b):
        return torch.log(1 + torch.exp(b * x)) / b
    

    def euler2fixedpt(self, x_initial, Nmax=100, Navg=80, dt=0.001, xtol=1e-5, xmin=1e-0):
        xmin = torch.tensor(xmin, device=self.device, requires_grad=False)

        avgStart = Nmax - Navg
        avg_sum = 0
        xvec = x_initial
        
        # res = []

        for _ in range(avgStart):  # Loop without taking average step size
            # dx = self.drdt_func(xvec) * dt
            self._update_mu_sigma(xvec)
            dx = self.T_inv * (self.phi() - xvec) * dt
            # xvec = xvec + self.T_inv * (self.phi() - xvec)
            xvec = xvec + dx
            # res.append(xvec[50].item())

        for _ in range(Navg):  # Loop whilst recording average step size
            self._update_mu_sigma(xvec)
            dx = self.T_inv * (self.phi() - xvec) * dt
            xvec = xvec + dx
            avg_sum = avg_sum + torch.abs(dx / torch.maximum(xmin, torch.abs(xvec)) ).max() / xtol
            # res.append(xvec[50].item())

        # plt.plot(res)
        # plt.show()

        return xvec, avg_sum / Navg


    # This is the input-output function (for mean-field spiking neurons) that you would use Max
    def phi(self):

        # Might need error handling for mu and sigma being None
        xp = (self.mu - self.Vr) / self.sigma
        xm = (self.mu - self.Vt) / self.sigma
        

        # rate = torch.zeros_like(xm, device=self.device) # dunno why we need this?
        xm_pos = self.sigmoid(xm * self.hardness)
        inds = self.sigmoid(-xm * self.hardness) * self.sigmoid(xp * self.hardness)
        
        xp1 = self.softplus(xp, self.hardness)
        xm1 = self.softplus(xm, self.hardness)
        
        #xm_pos = xm > 0
        # rate = (rate * (1 - xm_pos)) + (xm_pos / self.softplus(self.f_ricci(xp1) - self.f_ricci(xm1), self.hardness))
        rate = (xm_pos / self.softplus(self.f_ricci(xp1) - self.f_ricci(xm1), self.hardness))


        #inds = (xp > 0) & (xm <= 0)
        rate = (rate * (1 - inds)) + (inds / (self.f_ricci(xp1) + torch.exp(xm**2) * self.g_ricci(self.softplus(-xm, self.hardness))))
        
        rate = 1 / (self.tau_ref + 1 / rate)

        return rate / self.tau


    def f_ricci(self, x):
        z = x / (1 + x)
        return torch.log(2*x + 1) + (self.a[1] *(-z)**1 + self.a[2] *(-z)**2 + self.a[3] *(-z)**3
                                + self.a[4] *(-z)**4 + self.a[5] *(-z)**5 + self.a[6] *(-z)**6
                                + self.a[7] *(-z)**7 + self.a[8] *(-z)**8 + self.a[9] *(-z)**9)

    @staticmethod
    def g_ricci(x):

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



def training_loop(model, optimizer, Y, n=1000, device="cpu"):
    "Training loop for torch model."

    with open(f"log_run_{time.time()}.log", "w") as f:
        loss_function = MMDLossFunction(device=device)
        model.train()

        for i in range(n):
            optimizer.zero_grad()
            preds, avg_step = model()
            print("Computing loss...")
            loss = loss_function(preds, Y, avg_step)
            print("Computed loss: ", loss)
            print("Backwards...")
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


def training_loop_no_backwards(model, Y, n=1000, device="cpu"):
    "Training loop for torch model."

    with open(f"log_run_{time.time()}.log", "w") as f:
        loss_function = MMDLossFunction(device=device)
        model.train()

        for i in range(n):
            preds, avg_step = model()
            print("Computing loss...")
            loss = loss_function(preds, Y, avg_step)
            print("loss: ", loss)
            f.write(f"ITTER: {i + 1}  {preds.shape}\n")
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


def get_data(device="cpu"):
    df = pd.read_csv("./data/K-Data.csv")
    v1 = df.query("region == 'V1'")
    m = v1.m.unique()[2]
    v1 = v1[v1.m == m] # take for all mice later
    v1 = v1.copy()  # to prevent warning
    v1["mouse_unit"] = v1["m"] + "_" + v1["u"].astype(str)
    v1 = v1.groupby(["mouse_unit", "grat_orientation", "grat_contrast", "grat_spat_freq", "grat_phase"]).mean(numeric_only=True).reset_index()
    v1 = v1[["mouse_unit", "grat_orientation", "grat_contrast", "grat_spat_freq", "grat_phase", "response"]]

    unique_units = v1['mouse_unit'].unique()
    unique_orientation = v1['grat_orientation'].unique()
    unique_contrast = v1['grat_contrast'].unique()
    unique_spat_freq = v1['grat_spat_freq'].unique()
    unique_phase = v1['grat_phase'].unique()

    shape = (len(unique_units), len(unique_orientation), len(unique_contrast), len(unique_spat_freq), len(unique_phase))
    result_array = np.full(shape, np.nan)

    # Iterate through the DataFrame and fill the array
    for index, row in tqdm(v1.iterrows()):
        u_index = np.where(unique_units == row['mouse_unit'])[0][0]
        orientation_index = np.where(unique_orientation == row['grat_orientation'])[0][0]
        contrast_index = np.where(unique_contrast == row['grat_contrast'])[0][0]
        spat_freq_index = np.where(unique_spat_freq == row['grat_spat_freq'])[0][0]
        phase_index = np.where(unique_phase == row['grat_phase'])[0][0]
        result_array[u_index, orientation_index, contrast_index, spat_freq_index, phase_index] = row['response']

    result_array = np.mean(np.mean(result_array, axis=4), axis=3)
    result_array = result_array.transpose((0, 2, 1))
    result_array = result_array * 1000

    result_array = torch.tensor(result_array, device=device)
    return result_array




if __name__ == "__main__":
    
    GRAD = False
    print("Grad: ", GRAD)
    if torch.cuda.is_available():
        device = "cuda"
        print("Model will be created on GPU")
    else:
        device = "cpu"
        print("GPU not available. Model will be created on CPU.")

    result_array = get_data(device=device)

    # J_array = [0.69, 0.64, 0., -0.29] # Max log values
    # P_array = [-2.21, -2.21, -0.8, -0.8]
    # w_array = [3.46, 3.46, 3.46, 3.46]

    J_array = [0, -0.29, 0.69, -0.64] # Keen log values
    P_array = [-0.8, -2.21, -2.21, -0.8]
    w_array = [3.46, 3.46, 3.46, 3.46]

    if GRAD:
        model = NeuroNN(J_array, P_array, w_array, 4000, device=device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        training_loop(model, optimizer, result_array, device=device)
    else:
        model = NeuroNN(J_array, P_array, w_array, 10000, device=device, grad=False)
        training_loop_no_backwards(model, result_array, device=device)

    # https://towardsdatascience.com/how-to-use-pytorch-as-a-general-optimizer-a91cbf72a7fb
