import torch
from .main import Rodents


class NetworkExecuter(Rodents):
    def __init__(self, neuron_num, feed_forward_num=100, ratio=0.8, scaling_g=0.15, w_ff=30, sig_ext=5, device="cpu", plot_overtime=False, 
                 contrasts=[0, 0.0432773, 0.103411, 0.186966, 0.303066, 0.464386, 0.68854, 1.],
                 orientations=[0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]):
        super().__init__(neuron_num, ratio, device, feed_forward_num)

        self.orientations = orientations  # 12  # NOTE: we can reduce this for experimental runs, we can change this as we move closer to the optimal during optimisation
        self.contrasts = contrasts  # 8

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

    def __init__(self, neuron_num, feed_forward_num=100, ratio=0.8, scaling_g=0.15, w_ff=30, sig_ext=5, device="cpu", plot_overtime=False,
                 contrasts=[0, 0.0432773, 0.103411, 0.186966, 0.303066, 0.464386, 0.68854, 1.],
                 orientations=[0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]):
        super().__init__(neuron_num, feed_forward_num, ratio, scaling_g, w_ff, sig_ext, device, plot_overtime, contrasts, orientations)
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
        rate = rate.view(self.neuron_num, len(self.contrasts), len(self.orientations))
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


class NetworkExecuterWithSimplifiedFF(NetworkExecuterParallel):

    def __init__(self, neuron_num, feed_forward_num=100, ratio=0.8, scaling_g=0.15, w_ff=30, sig_ext=5, device="cpu", plot_overtime=False,
                 contrasts=[0, 0.0432773, 0.103411, 0.186966, 0.303066, 0.464386, 0.68854, 1.],
                 orientations=[0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]):
        super().__init__(neuron_num, feed_forward_num, ratio, scaling_g, w_ff, sig_ext, device, plot_overtime, contrasts, orientations)
        self.heter_ff = None


    def _stim_to_inputs(self):
        '''Set the inputs based on the contrast and orientation of the stimulus and heter_FF'''
        if self.heter_ff == None:
            raise ValueError("heter_ff is None.")
        input_mean = []
        mu_BL = self.Vt - 2.71 * self.sig_ext
        random_term = (1 + self.heter_ff * ((2 * torch.rand((len(self.orientations), self.neuron_num), device=self.device)) - 1))
        for contrast in self.contrasts:
            for i, orientation in enumerate(self.orientations):
                input_mean.append(random_term[i] * mu_BL * contrast * (self.Vt - self.Vr) * self.scaling_g * self._cric_gauss(orientation - self.pref, self.w_ff))  # NOTE: replace 20 with v_t - v_r
        self.input_mean = torch.stack(input_mean).T  # Dont forget to transpose back to get input for each constrast and orientation
        self.input_sd = self.sig_ext  # THIS DEFAULTS TO 5
        return self.input_mean, self.input_sd
    

    def update_heter_ff(self, heter_ff):
        self.heter_ff = self._sigmoid(heter_ff)
        