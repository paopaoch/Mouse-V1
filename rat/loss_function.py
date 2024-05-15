import torch

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
    def _optimised_kernel(U, V, W, sigma=1):
        return torch.exp(-1/(2*sigma**2) * (torch.diag(U)[:, None] + torch.diag(V)[None, :] - 2* W))

    
    @staticmethod
    def reshape_for_optimised(x: torch.Tensor):
        batch_size, dim1, dim2 = x.size()
        return torch.reshape(x, (batch_size, dim1 * dim2))
    

class MouseLossFunctionHomogeneous(MouseLossFunctionOptimised):

    @staticmethod
    def _transform_to_homogenous(tuning_curve): # (contrast, orientation)
        avg_tensor = torch.mean(tuning_curve, dim=1)
        return tuning_curve / avg_tensor.unsqueeze(1), avg_tensor


    def _transform_all_tuning_curves(self, tuning_curves): # (neurons, contrast, orientation)
        normalised_tuning_curves = []
        avg_tensors = []
        for tuning_curve in tuning_curves:  # TODO: remove loop
            normalised_tuning_curve, avg_tensor = self._transform_to_homogenous(tuning_curve)
            normalised_tuning_curves.append(normalised_tuning_curve.unsqueeze(0))
            avg_tensors.append(avg_tensor.unsqueeze(0))
        return torch.cat(normalised_tuning_curves, dim=0), torch.cat(avg_tensors, dim=0)


    def individual_MMD(self, x: torch.Tensor, y: torch.Tensor, sigma=1):
        G_x = x @ x.T
        G_y = y @ y.T
        G_xy = x @ y.T
        
        return torch.mean(self._optimised_kernel(G_x, G_x, G_x, sigma=sigma)) + torch.mean(self._optimised_kernel(G_y, G_y, G_y, sigma=sigma)) - torch.mean(2*self._optimised_kernel(G_x, G_y, G_xy, sigma=sigma))


    def MMD(self, x: torch.Tensor, y: torch.Tensor):

        x_normalised, x_avg_tensors = self._transform_all_tuning_curves(x)
        y_normalised, y_avg_tensors = self._transform_all_tuning_curves(y)

        x_normalised = self.reshape_for_optimised(x_normalised)
        y_normalised = self.reshape_for_optimised(y_normalised)

        MMD_normalised_tc = self.individual_MMD(x_normalised, y_normalised, sigma=1)
        MMD_avg = self.individual_MMD(x_avg_tensors, y_avg_tensors, sigma=30)  # TODO: Check what this sigma is

        return MMD_normalised_tc + MMD_avg  # Might need to add some weigting term