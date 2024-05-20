import torch
from rat import NetworkExecuterWithSimplifiedFF, WeightsGeneratorExact
from mouse_nes import make_torch_params, nes_multigaussian_optim
from utils.rodents_routine import create_directory_if_not_exists, get_device
from datetime import datetime
import numpy as np

if __name__ == "__main__":

    torch.manual_seed(96)

    dir_name = f"method_validation_{datetime.now()}"
    create_directory_if_not_exists(dir_name)

    desc = "NES with MMD"

    device = get_device("cuda:0")

    n = 10000

    executer = NetworkExecuterWithSimplifiedFF(n, device=device, scaling_g=0.15)

    # Create dataset
    J_array = [-0.9308613398652443, -2.0604571635972393, -0.30535063458645906, -1.802886963254238]
    P_array = [-1.493925025312256, 1.09861228866811, -1.493925025312256, 1.09861228866811]
    w_array = [-1.5314763709643886, -1.5314763709643886, -1.5314763709643886, -1.5314763709643886] 
    heter_ff = torch.tensor([-1.3862943611198906], device=device)

    J_array = torch.tensor(J_array, device=device)
    P_array = torch.tensor(P_array, device=device)
    w_array = torch.tensor(w_array, device=device)
    wg = WeightsGeneratorExact(J_array, P_array, w_array, n, device=device, forward_mode=True)
    W = wg.generate_weight_matrix()
    executer.update_heter_ff(heter_ff)
    tuning_curves, avg_step = executer.run_all_orientation_and_contrast(W)
    y_E, y_I = tuning_curves[:8000], tuning_curves[8000:]


    var_list = [0.5, 0.5, 0.5, 0.5,
                0.5, 0.5, 0.5, 0.5, 
                0.5, 0.5, 0.5, 0.5,
                0.5]

    for _ in range(11):
        file_name = f"{dir_name}/log_nes_run_{datetime.now()}"
        mean_array = np.random.rand(13) * 9 - 4.5
        mean_list: list = mean_array.tolist()
        mean, cov = make_torch_params(mean_list, var_list, device=device)

        nes_multigaussian_optim(mean, cov, 200, 13, y_E, y_I, device=device, neuron_num=n, desc=desc, trials=1, alpha=0.1, eta_delta=1, avg_step_weighting=0.1, stopping_criterion_step=0.0000001, adaptive_lr=False, file_name=file_name)