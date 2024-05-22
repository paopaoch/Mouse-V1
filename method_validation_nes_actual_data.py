import torch
from rat import NetworkExecuterWithSimplifiedFF, WeightsGeneratorExact, get_data
from mouse_nes import make_torch_params, nes_multigaussian_optim
from utils.rodents_routine import create_directory_if_not_exists, get_device
from datetime import datetime
import numpy as np
import time

if __name__ == "__main__":

    # torch.manual_seed(96)

    dir_name = f"method_validation_{time.time()}"
    create_directory_if_not_exists(dir_name)

    desc = "NES with MMD"

    device = get_device("cuda:0")

    n = 10000

    executer = NetworkExecuterWithSimplifiedFF(n, device=device, scaling_g=0.15)

    y_E, y_I = get_data(device=device)

    var_list = [0.5, 0.5, 0.5, 0.5,
                0.5, 0.5, 0.5, 0.5, 
                0.5, 0.5, 0.5, 0.5,
                0.5]

    for _ in range(30):
        file_name = f"{dir_name}/log_nes_run_{time.time()}"
        mean_array = np.random.rand(13) * 9 - 4.5
        mean_list: list = mean_array.tolist()
        mean, cov = make_torch_params(mean_list, var_list, device=device)

        nes_multigaussian_optim(mean, cov, 200, 12, y_E, y_I, device=device, neuron_num=n, desc=desc, trials=1, alpha=0.1, eta_delta=1, avg_step_weighting=0.1, stopping_criterion_step=0.0000001, adaptive_lr=False, file_name=file_name)