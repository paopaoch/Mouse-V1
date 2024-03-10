import torch
from mouse_nes import make_torch_params, nes_multigaussian_optim
import pickle
import numpy as np
import os
from time import time



def generate_mean_lists(start_mean_list, end_mean_list, num_samples=20):
    if len(start_mean_list) != len(end_mean_list):
        raise ValueError("Arrays must have the same length.")

    interpolated_arrays = np.linspace(start_mean_list, end_mean_list, num_samples + 2)[1:-1]
    return np.vstack([start_mean_list, *interpolated_arrays, end_mean_list])


if __name__ == "__main__":

    current_dir = "log_" + str(time())
    os.makedirs(f"./{current_dir}")

    if torch.cuda.is_available():
        device = "cuda:0"
        print("Model will be created on GPU")
    else:
        device = "cpu"
        print("GPU not available. Model will be created on CPU.")

    start_mean_list = [-4.054651081081644, -19.924301646902062, -0.0, -12.083112059245341, -6.591673732008658, 1.8571176252186712, -4.1588830833596715, 4.549042468104266, -167.03761889472233, -187.23627477210516, -143.08737747657977, -167.03761889472233]
    end_mean_list = [-4.054651081081644, -19.924301646902062, -0.0, -12.083112059245341, -6.591673732008658, 1.8571176252186712, -4.1588830833596715, 4.549042468104266, -127.03761889472233, -187.23627477210516, -143.08737747657977, -167.03761889472233]
    var_list = [5, 5, 5, 5, 
                1, 1, 1, 1, 
                250, 250, 250, 250]
    
    with open("data/data_1000_neurons2/responses.pkl", 'rb') as f:
        responses: torch.Tensor = pickle.load(f)
        y_E, y_I = responses[:800], responses[800:]
        responses = 0

    mean_lists = generate_mean_lists(start_mean_list, end_mean_list, num_samples=5)
    start_means = []
    end_means = []
    end_covs = []
    num_iters = []
    with open(f"./{current_dir}/metadata.log", "w") as f:
        
        f.write("#### Investigating deviation from ground truth ####\n\n")
        f.write("Investigating the deviation effect of the width values\n\n")
        _, start_cov = make_torch_params(start_mean_list, var_list, device=device)
        f.write(f"starting_cov: {start_cov}\n")
        f.flush()

        for i, mean_list in enumerate(mean_lists):
            desc = f"Vary w value iteration {i}"
            
            start_mean, start_cov = make_torch_params(mean_list, var_list, device=device)
            mean, cov, num_iter = nes_multigaussian_optim(start_mean, start_cov, 200, 12, y_E, y_I, device=device, neuron_num=1000, desc=desc,
                                                trials=1, alpha=0.1, eta_delta=1, avg_step_weighting=0.05, min_iter=20, 
                                                stopping_criterion_step=0.00001, stopping_criterion_tolerance=2, file_name=f"./{current_dir}/run_{i}.log")

            f.write(f"\n\n\n")
            f.write(f"-----------------------\n\n\n")
            f.write(f"ITERATION: {i}\n")
            f.write(f"starting_mean: {start_mean}\n")
            f.write(f"ending_mean: {mean}\n")
            f.write(f"ending_cov: {cov}\n")
            f.write(f"number of nes steps: {num_iter}\n")
            f.flush()

            start_means.append(start_mean)
            end_means.append(mean)
            end_covs.append(cov)
            num_iters.append(num_iter)

    output = {"start_means": start_means,
              "end_means": end_means,
              "end_covs": end_covs,
              "num_itters": num_iter}
    
    with open(f"./{current_dir}/results.pkl", "wb") as pf:
        pickle.dump(output, pf)