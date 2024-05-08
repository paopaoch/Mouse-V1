import torch
import time
from datetime import datetime
import sys
from tqdm import tqdm
from rat import MouseLossFunctionOptimised, WeightsGeneratorExact, NetworkExecuterParallel, get_data
import socket
import pickle


class MouseDataPoint:
    def __init__(self, loss: torch.Tensor, zk: torch.Tensor, prob: torch.Tensor, accepted=True):
        self.loss = loss
        self.zk = zk
        self.prob = prob
        self.accepted = accepted
    
    def get_loss(self):
        return self.loss.clone().detach()
    

    def get_zk(self):
        return self.zk.clone().detach()
    

    def get_prob(self):
        return self.prob.clone().detach()
    

    def update_prob(self, new_prob: torch.Tensor):
        self.prob = new_prob.clone().detach()


def make_torch_params(mean_list, var_list, device="cpu"):
    """Return a tensor with mean and a diagonal covariance matrix, TODO: add positive definite check"""
    mean_tensor = torch.tensor(mean_list, device=device, dtype=torch.float32)
    var_tensor = torch.diag(torch.tensor(var_list, device=device, dtype=torch.float32))
    return mean_tensor, var_tensor


def mean_to_params(mean):
    if len(mean) == 12:  # No feed forward
        return mean[0:4], mean[4:8], mean[8:12]
    elif len(mean) == 18:  # With feed forward
        return mean[0:6], mean[6:12], mean[12:18]
    else:
        raise IndexError(f"Expected an array of size 12 or 18 but found size {len(mean)}")


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


def calc_loss(trials,
             weights_generator: WeightsGeneratorExact, 
             network_executer: NetworkExecuterParallel, 
             loss_function: MouseLossFunctionOptimised,
             y_E, y_I, feed_forward=False):
    loss_sum = 0
    mmd_sum = 0
    for _ in range(trials):
        if feed_forward:
            weights_FF = weights_generator.generate_feed_forward_weight_matrix()
        else:
            weights_FF = None
        weights = weights_generator.generate_weight_matrix()
        preds, avg_step = network_executer.run_all_orientation_and_contrast(weights, weights_FF)
        preds_E = preds[:weights_generator.neuron_num_e]
        preds_I = preds[weights_generator.neuron_num_e:]
        trial_loss, trial_mmd_loss = loss_function.calculate_loss(preds_E, y_E, preds_I, y_I, avg_step)
        loss_sum += trial_loss
        mmd_sum += trial_mmd_loss
    return loss_sum / trials, mmd_sum / trials


def nes_multigaussian_optim(mean: torch.Tensor, cov: torch.Tensor, max_iter: int, samples_per_iter: int, y_E, y_I,
                            neuron_num=10000, feed_forward_num=100, eta_delta=1, eta_sigma=0.08, eta_B=0.08, 
                            eta_sigma_min=0.08, eta_sigma_max=1, eta_B_min=0.08, eta_B_max=1,
                            device="cpu", avg_step_weighting=0.002, desc="", alpha=0.6, trials=1, weights_valid_weighting=1e5,
                            min_iter=20, stopping_criterion_step=1e-5, stopping_criterion_tolerance=2, file_name=None, 
                            beta=0.2, adaptive_lr=False):
    start = time.time()

    # local variable setup
    alpha = torch.tensor(alpha, device=device)
    beta = torch.tensor(beta, device=device)
    J, P, w = mean_to_params(mean)
    if len(mean) == 18:
        feed_forward = True
    else:
        feed_forward = False
    
    # Init model and loss function
    loss_function = MouseLossFunctionOptimised(device=device, avg_step_weighting=avg_step_weighting)
    network_executer = NetworkExecuterParallel(neuron_num, device=device, feed_forward_num=feed_forward_num)
    weights_generator = WeightsGeneratorExact(J, P, w, neuron_num, feed_forward_num=feed_forward_num, device=device)
    weights_valid = weights_generator.validate_weight_matrix()

    # if weights_valid != torch.tensor(0, device=device):
    #     print("ERROR WEIGHT IS NOT VALID")
    #     return
    if file_name is None:
        file_name = f"log_nes_run_{time.time()}.log"
        
    with open(file_name, "w") as f:
        # write the metadata to log file
        f.write("#### Mouse V1 Project log file ####\n\n")
        f.write(f"Code ran on the {datetime.now()}\n\n")
        f.write(f"Device: {device}\n")
        f.write(f"OS: {sys.platform}\n")
        f.write(f"Machine: {socket.gethostname()}\n")
        f.write("Trainer type: xNES\n\n")
        f.write(f"{desc}\n\n")
        f.write("Metadata:\n")
        f.write(f"Number of neurons: {weights_generator.neuron_num}\n")
        f.write(f"Number of feedforward neurons: {weights_generator.feed_forward_num}\n")
        f.write(f"Number of Euler steps: {network_executer.Nmax}\n")
        f.write(f"Record average step after {network_executer.Navg} steps\n")
        f.write(f"Average step weighting: {avg_step_weighting}\n")
        f.write(f"Valid weight weighting: {weights_valid_weighting}\n\n")
        f.write(f"Max_number of xNES optimisation step: {max_iter}\n")
        f.write(f"Min_number of xNES optimisation step: {min_iter}\n")
        f.write(f"Stopping criterion step condition for xNES optimisation: {stopping_criterion_step}\n")
        f.write(f"Stopping criterion step tolerance: {stopping_criterion_tolerance}\n")
        f.write(f"Number of samples per optimisation step: {samples_per_iter}\n")
        f.write(f"Number of trials per full simulation: {trials}\n")
        f.write(f"Alpha for important mixing: {alpha}\n")
        f.write(f"Learning Rates: eta_delta={eta_delta}, eta_sigma={eta_sigma}, eta_B={eta_B}\n")
        f.write(f"adaptive_lr: {adaptive_lr}\n")
        f.write(f"beta: {beta}\n")
        f.write(f"---------------------------------------------------\n\n")
        f.write(f"Initial parameters\n")
        f.write(f"J\n")
        f.write(str(weights_generator.J_parameters))
        f.write("\n")
        f.write(f"P\n")
        f.write(str(weights_generator.P_parameters))
        f.write("\n")
        f.write(f"w\n")
        f.write(str(weights_generator.w_parameters))
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
        utilities = get_utilities(samples_per_iter, device=device)  # Utilities do not depend on the samples but depend on the number of samples, so is the same throughout
        mu_w = torch.sum(1/(utilities ** 2))
        pS = torch.zeros_like(B, device=device, dtype=torch.float32)
        gamma_theta = torch.tensor(0., device=device)
        prev_samples: list[MouseDataPoint] = []
        prev_mean = mean.clone().detach()
        prev_sigma = sigma.clone().detach()
        prev_B = B.clone().detach()

        avg_nes_step = 0
        nes_loss = []
        stopping_reached_count = 0
        prev_avg_nes_step = torch.tensor(10000, dtype=torch.float32, device=device)  # Set to be a very large number for stopping criterion
        for i in tqdm(range(max_iter)):
            f.write(f"ITERATION: {i}\n")

            samples = []
            losses = []
            current_samples = []
            rejected = 0
            accepted_loss = []

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
                    if prev_sample.accepted:
                        accepted_loss.append(prev_sample.get_loss())  # TODO: Seems to be a minor bug as when an invalid sample is selected, rejected does not increase

            print(f"Number of samples reused: {len(samples)}")
            f.write(f"Number of samples reused: {len(samples)}\n")

            J, P, w = mean_to_params(mean)
            weights_generator.set_parameters(J, P, w)
            mean_loss, mean_MMD_loss = calc_loss(trials, weights_generator, network_executer, loss_function, y_E, y_I, feed_forward=feed_forward)
            
            print("current_mean_loss: ", mean_loss, mean_MMD_loss)
            print("mean: ", mean)
            f.write(f"Mean: {mean}\n")
            f.write(f"Mean loss: {mean_loss}\n")
            f.write(f"MMD Mean loss: {mean_MMD_loss}\n")

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
                    weights_generator.set_parameters(J, P, w)
                    weights_valid = weights_generator.validate_weight_matrix()
                    if weights_valid == torch.tensor(0, device=device):
                        current_loss, _ = calc_loss(trials, weights_generator, network_executer, loss_function, y_E, y_I, feed_forward=feed_forward)
                        accepted_loss.append(current_loss.clone().detach())
                        accepted = True
                    else:
                        current_loss = weights_valid_weighting * weights_valid
                        rejected += 1
                        accepted = False

                    samples.append(sample)
                    losses.append(current_loss.clone().detach())

                    data_point = MouseDataPoint(current_loss.clone().detach(), 
                                                zk.clone().detach(), 
                                                prob.clone().detach(),
                                                accepted=accepted)
                    
                    current_samples.append(data_point)

            if len(accepted_loss) == 0:
                accepted_loss.append(torch.tensor(0, device=device, dtype=torch.float32))

            loss_sorted, samples_sorted = sort_two_arrays(losses, samples, device=device)
            accepted_loss_tensor = torch.tensor(accepted_loss, device=device)

            avg_loss = torch.mean(loss_sorted)
            min_loss = torch.min(loss_sorted)
            max_loss = torch.max(loss_sorted)
            print("Min loss", min_loss)
            print("Avg loss", avg_loss)
            f.write(f"Avg loss {avg_loss}\n")
            f.write(f"Min loss {min_loss}\n")
            f.write(f"Max loss {max_loss}\n")
            f.write(f"Avg_accepted loss {torch.mean(accepted_loss_tensor)}\n")
            f.write(f"Min_accepted loss {torch.min(accepted_loss_tensor)}\n")
            f.write(f"Max_accepted loss {torch.max(accepted_loss_tensor)}\n")
            f.write(f"Rejected {rejected}\n")
            f.write("\n\n")

            # Compute natural gradients
            grad_delta = (samples_sorted.permute(1, 0) * utilities).permute(1, 0).sum(dim=(0))
            grad_M = torch.zeros(size=(len(samples_sorted[0]), len(samples_sorted[0])), device=device)
            for k, sample in enumerate(samples_sorted):
                grad_M += ((sample * sample.t()) - torch.eye(len(sample), device=device)) * utilities[k]

            grad_sigma = torch.trace(grad_M) / d
            grad_B = torch.trace(grad_M) - grad_sigma * torch.eye(len(grad_M), device=device)
            
            # initialise and calc delta for learning rate adaptation
            if adaptive_lr:  # check all of the 
                delta_m = - mean.clone().detach()
                delta_cov = - (sigma ** 2) * B.T @ B
                cov = (sigma ** 2) * B.T @ B
                e, v = torch.linalg.eigh(cov)
                diag_sqrt_eig = torch.diag(torch.sqrt(e))
                inv_sqrt_cov: torch.Tensor = v.T @ torch.linalg.inv(diag_sqrt_eig) @ v

            # Gradient descent step
            mean = mean + eta_delta * sigma * B @ grad_delta
            sigma = sigma * torch.exp((eta_sigma / 2) * grad_sigma)
            B = B * torch.exp((eta_B / 2) * grad_B)
            f.flush()

            prev_samples = current_samples.copy()
            
            # adaptive learning rate
            if adaptive_lr:
                f.write(f"eta_sigma: {eta_sigma}\n")
                f.write(f"eta_B: {eta_B}\n")
                f.write("\n\n")
                f.flush()

                delta_m += mean
                delta_cov += (sigma ** 2) * B.T @ B
                approx = (1 / mu_w) * (eta_B**2 / 2) * (1 + 4*eta_sigma**2/(d*mu_w)) * (d**2 + d - 2) + eta_sigma  # equation 9 in the paper

                new_cov = (sigma ** 2) * B.T @ B
                normalised_new_cov = inv_sqrt_cov @ (new_cov @ inv_sqrt_cov)
                pS = (1 - beta) * pS + torch.sqrt(beta * (2 - beta)) / torch.sqrt(approx) * (normalised_new_cov - torch.eye(d, device=device))
                square_ptheta_norm = torch.trace(pS @ pS) / 2

                gamma_theta = (1 - beta)**2 * gamma_theta + beta * (2 - beta)

                eta_sigma = eta_sigma * torch.exp(beta * (square_ptheta_norm / alpha - gamma_theta))
                eta_sigma = min(max(eta_sigma, eta_sigma_min), eta_sigma_max)

                eta_B = eta_B * torch.exp(beta * (square_ptheta_norm / alpha - gamma_theta))
                eta_B = min(max(eta_B, eta_B_min), eta_B_max)

            # Stopping criterion
            nes_loss.append(mean_loss)
            avg_nes_step = torch.sum(torch.tensor(nes_loss[-10:], device=device)) / len(nes_loss[-10:])
            if i > min_iter and (prev_avg_nes_step - avg_nes_step) < stopping_criterion_step and rejected == 0:
                stopping_reached_count += 1
                if stopping_reached_count == stopping_criterion_tolerance:
                    f.write(f"\n\nEarly stopping {avg_nes_step}\n\n")
                    f.flush()
                    break
            else:
                stopping_reached_count = 0
            prev_avg_nes_step = avg_nes_step.clone().detach()

            if torch.isnan(mean).any().item():
                break

        # Get back parameters
        A_optimised = B * sigma
        cov_optimised = A_optimised.t() @ A_optimised

        J, P, w = mean_to_params(mean)
        weights_generator.set_parameters(J, P, w)
        mean_loss, mean_MMD_loss = calc_loss(trials, weights_generator, network_executer, loss_function, y_E, y_I, feed_forward=feed_forward)

        end = time.time()

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
        f.write(f"time taken: {end - start}\n")
        f.write(f"number of iterations: {i + 1}\n")
        f.write(f"average time per iter: {(end - start) / (i + 1)}\n")
        f.flush()

    return mean, cov_optimised, i


if __name__ == "__main__":

    torch.manual_seed(69)

    desc = "NES with JSD"

    if torch.cuda.is_available():
        device = "cuda:0"
        print("Model will be created on GPU")
    else:
        device = "cpu"
        print("GPU not available. Model will be created on CPU.")

    mean_list = [-1.7346010553881064, -2.586689344097943, -1.3862943611198906, -3.1780538303479458, -1.265666373331276, -0.6190392084062235, -1.265666373331276, -0.6190392084062235, -1.0986122886681098, -1.0986122886681098, -1.0986122886681098, -1.0986122886681098]
    n = 1000

    executer = NetworkExecuterParallel(n, device=device, scaling_g=0.15)

    # Create dataset
    J_array = [-2.059459853260332, -3.0504048076264896, -1.5877549090278045, -2.813481385641024]  # n = 1000
    P_array = [-2.0907410969337694, -0.20067069546215124, -2.0907410969337694, -0.20067069546215124]
    w_array = [-1.5314763709643886, -1.5314763709643886, -1.5314763709643886, -1.5314763709643886] 

    J_array = torch.tensor(J_array, device=device)
    P_array = torch.tensor(P_array, device=device)
    w_array = torch.tensor(w_array, device=device)
    wg = WeightsGeneratorExact(J_array, P_array, w_array, n, device=device, forward_mode=True)
    W = wg.generate_weight_matrix()
    tuning_curves, avg_step = executer.run_all_orientation_and_contrast(W)
    y_E, y_I = tuning_curves[:800], tuning_curves[800:]


    var_list = [0.5, 0.5, 0.5, 0.5,
                0.5, 0.5, 0.5, 0.5, 
                0.5, 0.5, 0.5, 0.5,]

    mean, cov = make_torch_params(mean_list, var_list, device=device)

    nes_multigaussian_optim(mean, cov, 200, 24, y_E, y_I, device=device, neuron_num=n, desc=desc, trials=1, alpha=0.1, eta_delta=1, avg_step_weighting=0.1, stopping_criterion_step=0.0000001, adaptive_lr=False)