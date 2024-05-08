import torch
from rat import NetworkExecuterParallel, WeightsGeneratorExact
from mouse_nes import make_torch_params, nes_multigaussian_optim
from utils.rodents_routine import create_directory_if_not_exists
from datetime import datetime
import numpy as np

if __name__ == "__main__":

    # torch.manual_seed(69)

    dir_name = f"method_validation_{datetime.now()}"
    create_directory_if_not_exists(dir_name)

    desc = "NES with JSD"

    if torch.cuda.is_available():
        device = "cuda:0"
        print("Model will be created on GPU")
    else:
        device = "cpu"
        print("GPU not available. Model will be created on CPU.")

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

    for _ in range(11):
        file_name = f"{dir_name}/log_nes_run_{datetime.now()}"
        mean_array = np.random.rand(12) * 9 - 4.5
        mean_list = mean_array.tolist()
        mean, cov = make_torch_params(mean_list, var_list, device=device)

        nes_multigaussian_optim(mean, cov, 200, 24, y_E, y_I, device=device, neuron_num=n, desc=desc, trials=1, alpha=0.1, eta_delta=1, avg_step_weighting=0.1, stopping_criterion_step=0.0000001, adaptive_lr=False, file_name=file_name)