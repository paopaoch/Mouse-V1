from rat import WeightsGeneratorExact, NetworkExecuterParallel
from mouse_nes import nes_multigaussian_optim, make_torch_params
import torch
import numpy as np
import pickle
import os
from time import time
from parameter_tester import mean_list_to_values


def get_deviated_mean(mean, cov, device="cpu"):
    mean = torch.tensor(mean, device=device)
    covariance_matrix = torch.tensor(cov, device=device)
    mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix)
    num_samples = 10
    samples = mvn.sample((num_samples,))
    return samples


def generate_varying_lists(start_list, end_list, num_samples=20):
    if len(start_list) != len(end_list):
        raise ValueError("Arrays must have the same length.")

    interpolated_arrays = np.linspace(start_list, end_list, num_samples + 2)[1:-1]
    return np.vstack([start_list, *interpolated_arrays, end_list])

if __name__ == "__main__":
    current_dir = "log_" + str(time())
    os.makedirs(f"./{current_dir}")

    if torch.cuda.is_available():
        device = "cuda:0"
        print("Model will be created on GPU")
    else:
        device = "cpu"
        print("GPU not available. Model will be created on CPU.")

    with open("./data/data_1000_neuron3/responses.pkl", 'rb') as f:
        responses: torch.Tensor = pickle.load(f)
        responses = responses.to(device)
        y_E, y_I = responses[:800], responses[800:]
        responses = 0

    mean_list = [-20.907410969337693, -30.550488507104102, -15.85627263740382, -28.060150147989074, 
                -1.2163953243244932, 8.833316937499324, -1.2163953243244932, 8.833316937499324, 
                -138.44395575681614, -138.44395575681614, -138.44395575681614, -138.44395575681614]

    ground_truth_vals = torch.tensor([11, 4.5, 17, 5.7, 0.4, 0.95, 0.4, 0.95, 57, 57, 57, 57]
                                     , device=device, dtype=torch.float32)

    start_var_list = [0.01, 0.01, 0.01, 0.01,
                        0.01, 0.01, 0.01, 0.01,
                        0.01, 0.01, 0.01, 0.01]

    end_var_list = [5, 5, 5, 5, 
                    1, 1, 1, 1, 
                    250, 250, 250, 250]
    
    mean, cov_nes = make_torch_params(mean_list, end_var_list, device=device)

    var_lists = generate_varying_lists(start_var_list, end_var_list, 20)

    desc = "Investigating deviation from mean"

    with open(f"./{current_dir}/metadata.log", "w") as f:
        f.write("#####LOG FOR DEVIATION#####\n\n")
        f.write(f"\n\n")
        f.flush()

        for i, var_list in enumerate(var_lists):
            _, cov = make_torch_params(mean_list, var_list)
            mean_tensors = get_deviated_mean(mean, cov, device=device)
            sum_diff_from_truth = 0
            sum_diff_from_start = 0
            sum_num_iter = 0
            for j, mean in enumerate(mean_tensors):
                mean_optimised, cov_optimised, num_iter = nes_multigaussian_optim(mean, cov_nes, 200, 12, y_E, y_I, 
                                                                                device=device, neuron_num=1000, 
                                                                                desc=desc, trials=1, alpha=0.05, 
                                                                                avg_step_weighting=0.1, 
                                                                                stopping_criterion_step=0.00001,
                                                                                file_name=f"./{current_dir}/run_{i}_{j}.log")

                start_mean_vals = torch.tensor(mean_list_to_values(mean), device=device, dtype=torch.float32)
                mean_optimised_vals = torch.tensor(mean_list_to_values(mean_optimised), device=device, dtype=torch.float32)
                diff_from_truth = torch.abs(ground_truth_vals - mean_optimised_vals)
                diff_from_start = torch.abs(start_mean_vals-mean_optimised_vals)

                sum_diff_from_truth += diff_from_truth
                sum_diff_from_start += diff_from_start
                sum_num_iter += num_iter

            avg_diff_from_truth = sum_diff_from_truth / len(mean_tensors)
            avg_diff_from_start = sum_diff_from_start / len(mean_tensors)
            avg_num_iter = sum_num_iter / len(mean_tensors)

            f.write(f"Iteration {i}\n")
            f.write(f"Variance: {var_list}\n")
            f.write(f"avg_num_iter: {avg_num_iter}\n")
            f.write(f"avg_diff_from_truth: {avg_diff_from_truth}\n")  # Expect this to stay somewhat the same and low if the model is good
            f.write(f"avg_diff_from_start: {avg_diff_from_start}\n")  # Expect this to be higher
            f.write(f"\n\n")
            f.flush()