import torch
from .main import Rodents


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
        self.J_scale = 40

        self.P_steep = 1
        self.P_scale = 0.6

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
    

class OSDependentWeightsGenerator(WeightsGenerator):
    def _get_sub_weight_matrix(self, diff: torch.Tensor, index: int):
        J_single = self._sigmoid(self.J_parameters[index], self.J_steep, self.J_scale) / torch.sqrt(torch.tensor(diff.shape[1]))
        circ_matrix = self._cric_gauss(diff, self._sigmoid(self.w_parameters[index], self.w_steep, self.w_scale))
        return J_single * circ_matrix

