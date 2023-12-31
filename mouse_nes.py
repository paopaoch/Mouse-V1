import torch
import time
from datetime import datetime
import sys
from tqdm import tqdm
from mouse import MMDLossFunction, NeuroNN
from mouse_trainer_functions import get_data


class MouseDataPoint:
    def __init__(self, loss: torch.Tensor, zk: torch.Tensor, prob: torch.Tensor):
        self.loss = loss
        self.zk = zk
        self.prob = prob
    
    def get_loss(self):
        return self.loss.clone().detach()
    

    def get_zk(self):
        return self.zk.clone().detach()
    

    def get_prob(self):
        return self.prob.clone().detach()
    

    def update_prob(self, new_prob: torch.Tensor):
        self.prob = new_prob.clone().detach()


def make_torch_params(mean_list, var_list, device="cpu"):
    """Return a tensor with mean and a diagonal covariance matrix, TODO: add shape checking, positive definite check"""
    # Mean
    d = len(var_list)
    mean_tensor = torch.tensor(mean_list, device=device, dtype=torch.float32)
    var_tensor = torch.diag(torch.tensor(var_list, device=device, dtype=torch.float32)) + torch.ones((d, d), device=device) * 0.1
    return mean_tensor, var_tensor


def mean_to_params(mean):
    return mean[0:4], mean[4:8], mean[8:12]


def get_utilities(length: int, device="cpu"):  # samples are sorted in ascending order as we want lower loss
    lamb = torch.tensor(length, device=device)
    log_lamb = torch.log(lamb/2 + 1)
    denominator = torch.tensor(0, dtype=torch.float32, device=device)
    numerators = []
    for i in range(1, lamb + 1):
        value = max(torch.tensor(0, dtype=torch.float32, device=device), log_lamb - torch.log(torch.tensor(i, dtype=torch.float32, device=device)))
        numerators.append(value)
        denominator += value
    return torch.stack(numerators) / denominator - (1 / lamb)


def sort_two_arrays(losses: list, samples: list, device="cpu"):  # sort according to array1
    combined_arrays = zip(losses, samples)
    sorted_combined = sorted(combined_arrays, key=lambda x: x[0])
    sorted_losses, sorted_samples = zip(*sorted_combined)
    return torch.tensor(sorted_losses, device=device), torch.stack(sorted_samples)  # WARNING: Very prone to error, double check this


def nes_multigaussian_optim(mean: torch.Tensor, cov: torch.Tensor, max_iter: int, samples_per_iter: int, Y, 
                            neuron_num=10000, eta_delta=1, eta_sigma=0.08, eta_B=0.08, 
                            device="cpu", avg_step_weighting=0.002, desc="", alpha=torch.tensor(0.1)):
    # Init model and loss function
    J, P, w = mean_to_params(mean)
    model = NeuroNN(J, P, w, neuron_num, device=device, grad=False)
    loss_function = MMDLossFunction(device=device, avg_step_weighting=avg_step_weighting)

    with open(f"log_nes_run_{time.time()}.log", "w") as f:
        # write the metadata to log file
        f.write("#### Mouse V1 Project log file ####\n\n")
        f.write(f"Code ran on the {datetime.now()}\n\n")
        f.write(f"Device: {device}\n")
        f.write(f"OS: {sys.platform}\n")
        f.write("Trainer type: xNES\n\n")
        f.write(f"{desc}\n\n")
        f.write("Metadata:\n")
        f.write(f"Number of neurons: {model.neuron_num}\n")
        f.write(f"Number of Euler steps: {model.Nmax}\n")
        f.write(f"Record average step after {model.Navg} steps\n")
        f.write(f"Average step weighting: {avg_step_weighting}\n")
        f.write(f"Number of xNES optimisation step: {max_iter}\n")
        f.write(f"Number of of samples per optimisation step: {samples_per_iter}\n")
        f.write(f"Learning Rates: eta_delta={eta_delta}, eta_sigma={eta_sigma}, eta_B={eta_B}\n")
        f.write(f"---------------------------------------------------\n\n")
        f.write(f"Initial parameters\n")
        f.write(f"J\n")
        f.write(str(model.j_hyperparameter))
        f.write("\n")
        f.write(f"P\n")
        f.write(str(model.p_hyperparameter))
        f.write("\n")
        f.write(f"w\n")
        f.write(str(model.w_hyperparameter))
        f.write("\n\n")
        f.write("Covariance Matrix\n")
        f.write(str(cov))
        f.write("\n\n")
        f.write(f"---------------------------------------------------\n\n\n")
        f.flush()

        d = len(mean)  # dimensions
        # Get cov = A^TA
        A_T: torch.Tensor = torch.linalg.cholesky(cov)
        A = A_T.t()

        # Get sigma
        sigma = torch.pow(torch.det(A), 1/d)
        B = A / sigma

        # initialise a multigaussian dist for sample
        mean_zeros = torch.zeros(d, device=device)
        cov_iden = torch.eye(d, device=device)
        multivariate_normal = torch.distributions.MultivariateNormal(mean_zeros, cov_iden)
        utilities = get_utilities(samples_per_iter, device=device)  # Utilities do not depend on the samples but depend on the number of samples, so fixed throughout
        prev_samples: list[MouseDataPoint] = []
        prev_mean = mean.clone().detach()
        prev_sigma = sigma.clone().detach()
        prev_B = B.clone().detach()

        for i in tqdm(range(max_iter)):
            f.write(f"ITERATION: {i}\n")

            samples = []
            losses = []
            current_samples = []

            # Important mixing
            for prev_sample in prev_samples:
                sk = torch.inverse(B.t()) @ ((prev_sample.get_zk() - mean) / sigma)
                prob = torch.exp(multivariate_normal.log_prob(sk))
                p = torch.minimum(torch.tensor(1, device=device), (1 - alpha) * (prob / prev_sample.get_prob()))
                accept_p = torch.rand(1)[0]
                if accept_p < p:
                    samples.append(sk)
                    losses.append(prev_sample.get_loss())
                    prev_sample.update_prob(p)
                    current_samples.append(prev_sample)

            print(f"Number of samples reused: {len(samples)}")
            f.write(f"Number of samples reused: {len(samples)}\n")

            J, P, w = mean_to_params(mean)
            preds, avg_step = model()
            mean_loss, mean_MMD_loss = loss_function(preds, Y, avg_step)
            print("current_mean_loss: ", mean_loss, mean_MMD_loss)
            print("mean: ", mean)
            f.write(f"Mean: {mean}\n")
            f.write(f"Mean loss: {mean_loss}\n")

            while len(samples) < samples_per_iter:  # Idealy, this could be done in parallel
                sample = multivariate_normal.sample((1,)).flatten()
                sample.to(device=device)
                prob = torch.exp(multivariate_normal.log_prob(sample))

                zk = mean + sigma * (B.t() @ sample)
                
                previous_sk = torch.inverse(prev_B.t()) @ ((zk - prev_mean) / prev_sigma)
                previous_prob = torch.exp(multivariate_normal.log_prob(previous_sk))
                p = torch.maximum(alpha, 1 - prob / previous_prob)
                accept_p = torch.rand(1)[0]
                if accept_p < p:
                    J, P, w = mean_to_params(zk)
                    model.set_parameters(J, P, w)
                    preds, avg_step = model()
                    current_loss, _ = loss_function(preds, Y, avg_step)

                    samples.append(sample)
                    losses.append(current_loss.clone().detach())

                    data_point = MouseDataPoint(current_loss.clone().detach(), 
                                                zk.clone().detach(), 
                                                prob.clone().detach())
                    current_samples.append(data_point)

            loss_sorted, samples_sorted = sort_two_arrays(losses, samples, device=device)
            
            avg_loss = torch.mean(loss_sorted)
            min_loss = torch.min(loss_sorted)
            print("Min loss", min_loss)
            print("Avg loss", avg_loss)
            f.write(f"Avg loss {avg_loss}\n")
            f.write(f"Min loss {min_loss}\n")
            f.write("\n\n\n")

            # Compute gradients
            grad_delta = (samples_sorted.permute(1, 0) * utilities).permute(1, 0).sum(dim=(0))
            grad_M = torch.zeros(size=(len(samples_sorted[0]), len(samples_sorted[0])), device=device)  # does not make sense to condense this
            for k, sample in enumerate(samples_sorted):
                grad_M += ((sample * sample.t()) - torch.eye(len(sample), device=device)) * utilities[k]

            grad_sigma = torch.trace(grad_M) / d
            grad_B = torch.trace(grad_M) - grad_sigma * torch.eye(len(grad_M), device=device)

            # Update parameters
            mean = mean + eta_delta * sigma * B @ grad_delta
            sigma = sigma * torch.exp((eta_sigma / 2) * grad_sigma)
            B = B * torch.exp((eta_B / 2) * grad_B)
            f.flush()

            prev_samples = current_samples.copy()

        # Get back parameters
        A_optimised = B * sigma
        cov_optimised = A_optimised.t() @ A_optimised

        J, P, w = mean_to_params(mean)
        preds, avg_step = model()
        mean_loss, mean_MMD_loss = loss_function(preds, Y, avg_step)

        f.write(f"---------------------------------------------------\n\n\n")
        f.write("Final loss and MMD loss:\n")
        f.write(str(mean_loss))
        f.write("\n")
        f.write(str(mean_MMD_loss))
        f.write("\n")
        f.write("Final mean:\n")
        f.write(str(mean))
        f.write("\n\n")
        f.write("Final covariance matrix:\n")
        f.write(str(cov_optimised))
        f.write("\n\n")
        f.flush()

    return mean, cov_optimised


if __name__ == "__main__":

    desc = "Second run. Fixed some bugs, the covariance matrix values were too big and is why it did not converge."

    if torch.cuda.is_available():
        device = "cuda"
        print("Model will be created on GPU")
    else:
        device = "cpu"
        print("GPU not available. Model will be created on CPU.")

    mean_list = [-4.39, -5.865, 0, -7.78,
                 -1.22, -6.592, -6.592, -1.22,
                 -45.94, -45.94, -45.94, -45.94]
    
    var_list = [0.5, 0.5, 0.5, 0.5, 
                0.5, 0.5, 0.5, 0.5, 
                0.5, 0.5, 0.5, 0.5]
    
    mean, cov = make_torch_params(mean_list, var_list, device=device)

    Y = get_data(device=device)

    print(nes_multigaussian_optim(mean, cov, 80, 12, Y, device=device, neuron_num=2000, desc=desc))
