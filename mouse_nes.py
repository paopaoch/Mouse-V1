import torch
import random
from time import time
from tqdm import tqdm
from mouse import MMDLossFunction, NeuroNN
from mouse_trainer_functions import get_data

# TODO: Make this a class

def make_torch_params(mean_list, var_list, device="cpu"):
    """Return a tensor with mean and a diagonal covariance matrix, TODO: add shape checking, positive definite check"""
    # Mean
    d = len(var_list)
    mean_tensor = torch.tensor(mean_list, device=device, dtype=torch.float32)
    var_tensor = torch.diag(torch.tensor(var_list, device=device, dtype=torch.float32)) + torch.ones((d, d), device=device) * 0.1
    return mean_tensor, var_tensor


def mean_to_params(mean):
    return mean[0:4], mean[4:8], mean[8:12]


def get_utilities(samples, device="cpu"):  # samples are sorted in ascending order as we want lower loss
    lamb = torch.tensor(len(samples), device=device)
    log_lamb = torch.log(lamb/2 + 1)
    denominator = torch.tensor(0, dtype=torch.float32, device=device)
    numerators = []
    for i in range(1, lamb + 1):
        value = max(torch.tensor(0, dtype=torch.float32, device=device), log_lamb - torch.log(torch.tensor(i, dtype=torch.float32, device=device)))
        numerators.append(value)
        denominator += value
    return torch.stack(numerators) / denominator - (1 / lamb)


def sort_two_arrays(array1, array2, device="cpu"):  # sort according to array1
    combined_arrays = zip(array1, array2)
    sorted_combined = sorted(combined_arrays, key=lambda x: x[0])
    sorted_array1, sorted_array2 = zip(*sorted_combined)
    return torch.tensor(sorted_array1, device=device), torch.stack(sorted_array2)


def nes_multigaussian_optim(mean, cov, max_iter, samples_per_iter, Y, eta_delta=0.01, eta_sigma=0.01, eta_B=0.01, device="cpu"):
    
    # Init model and loss function
    J, P, w = mean_to_params(mean)
    model = NeuroNN(J, P, w, 1000, device=device, grad=False)
    loss_function = MMDLossFunction(device=device)


    d = len(mean)  # dimensions
    # Get cov = A^TA
    A_T: torch.Tensor = torch.linalg.cholesky(cov)
    A = A_T.t()

    # Get sigma
    sigma = torch.pow(torch.det(A), 1/d)
    B = A / sigma

    # initialise a multigaussian dist for sample
    mean_zeros = torch.zeros(d)
    cov_iden = torch.eye(d)
    multivariate_normal = torch.distributions.MultivariateNormal(mean_zeros, cov_iden)

    for _ in tqdm(range(max_iter)):
        samples = multivariate_normal.sample((samples_per_iter,), device=device)
        losses = []
        J, P, w = mean_to_params(mean)
        preds, avg_step = model()
        mean_loss, mean_MMD_loss = loss_function(preds, Y, avg_step)
        print("current_loss: ", mean_loss, mean_MMD_loss)
        for k in range(samples_per_iter):
            zk = mean + sigma * (B.t() @ samples[k])

            J, P, w = mean_to_params(zk)
            model.set_parameters(J, P, w)
            preds, avg_step = model()
            current_loss, MMD_loss = loss_function(preds, Y, avg_step)
            losses.append(current_loss.clone().detach())

        loss_sorted, samples_sorted = sort_two_arrays(losses, samples, device=device)
        utilities = get_utilities(loss_sorted, device=device)

        # Compute gradients
        grad_delta = (samples_sorted.permute(1, 0) * utilities).permute(1, 0).sum(dim=(0))
        grad_M = torch.zeros(size=(len(samples_sorted[0]), len(samples_sorted[0])), device=device)  # does not make sense to condense this
        for k, sample in enumerate(samples_sorted):
            grad_M += ((sample * sample.t()) - torch.eye(len(sample))) * utilities[k]

        grad_sigma = torch.trace(grad_M) / d
        grad_B = torch.trace(grad_M) - grad_sigma * torch.eye(len(grad_M), device=device)

        # Update parameters
        mean = mean + eta_delta * grad_delta
        sigma = sigma * torch.exp((eta_sigma / 2) * grad_sigma)
        B = B * torch.exp((eta_B / 2) * grad_B)

    # Get back parameters
    A_optimised = B * sigma
    cov_optimised = A_optimised.t() @ A_optimised

    return mean, cov_optimised


if __name__ == "__main__":


    if torch.cuda.is_available():
        device = "cuda"
        print("Model will be created on GPU")
    else:
        device = "cpu"
        print("GPU not available. Model will be created on CPU.")

    mean_list = [-4.39, -5.865, 0, -7.78,
                 -1.22, -6.592, -6.592, -1.22,
                 -12.25, -12.25, -12.25, -12.25]
    
    var_list = [5, 5, 5, 5, 
                5, 5, 5, 5, 
                1, 1, 1, 1]
    
    mean, cov = make_torch_params(mean_list, var_list, device=device)

    Y = get_data(device=device)

    print(nes_multigaussian_optim(mean, cov, 1000, 36, Y, device=device))
