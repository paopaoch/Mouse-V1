import torch
from rat import WeightsGenerator, NetworkExecuterParallel, MouseLossFunctionOptimised
import time
import os
import sys
import socket
from datetime import datetime


def round_1D_tensor_to_list(a, decimals=6):
    """
    Round each element of a tensor to the specified number of decimal places
    and return the result as a list of rounded values.
    
    Args:
        a (torch.Tensor): Input tensor of arbitrary length.
        decimals (int): Number of decimal places to round to.
        
    Returns:
        list: List of rounded values.
    """
    rounded_a = torch.round(a * (10 ** decimals)) / (10 ** decimals)
    rounded_list = [round(elem.item(), decimals) for elem in rounded_a]
    
    return rounded_list


def create_directory_if_not_exists(directory):
    """
    Create a directory if it does not already exist.

    Args:
        directory (str): Path of the directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
    else:
        print(f"Directory '{directory}' already exists.")


if __name__ == "__main__":

    directory_name = f"method_val_log_{time.time()}"
    create_directory_if_not_exists(directory_name)

    desc = "Method validation"

    if torch.cuda.is_available():
        device = "cuda:1"
        print("Model will be created on GPU")
    else:
        device = "cpu"
        print("GPU not available. Model will be created on CPU.")
    
    restart_num = 30

    executer = NetworkExecuterParallel(1000, device=device, scaling_g=0.15)
    loss_function = MouseLossFunctionOptimised(device=device)

    # Create dataset
    J_array = [-2.059459853260332, -3.0504048076264896, -1.5877549090278045, -2.813481385641024]  # n = 1000
    P_array = [-2.0907410969337694, -0.20067069546215124, -2.0907410969337694, -0.20067069546215124]
    w_array = [-1.5314763709643886, -1.5314763709643886, -1.5314763709643886, -1.5314763709643886] 

    J_array = torch.tensor(J_array, device= device)
    P_array = torch.tensor(P_array, device= device)
    w_array = torch.tensor(w_array, device= device)
    wg = WeightsGenerator(J_array, P_array, w_array, 1000, device=device, forward_mode=True)
    W = wg.generate_weight_matrix()
    tuning_curves, avg_step = executer.run_all_orientation_and_contrast(W)
    y_E, y_I = tuning_curves[:800], tuning_curves[800:]

    with open(f"{directory_name}/metadata.log", "w") as f:
        f.write("#####METHOD VALIDATION#####\n")
        f.write(f"Code ran on the {datetime.now()}\n\n")
        f.write(f"Device: {device}\n")
        f.write(f"OS: {sys.platform}\n")
        f.write(f"Machine: {socket.gethostname()}\n")
        f.write(f"{desc}\n\n")
        f.write(f"{round_1D_tensor_to_list(J_array)}\n")
        f.write(f"{round_1D_tensor_to_list(P_array)}\n")
        f.write(f"{round_1D_tensor_to_list(w_array)}\n")

    for _ in range(restart_num):
        J_array = torch.randn((4))
        P_array = torch.randn((4))
        w_array = torch.randn((4))
        # J_array = [-2.059459853260332, -3.0504048076264896, -1.5877549090278045, -2.813481385641024]  # n = 1000
        # P_array = [-2.0907410969337694, -0.20067069546215124, -2.0907410969337694, -0.20067069546215124]
        # w_array = [-1.5314763709643886, -1.5314763709643886, -1.5314763709643886, -1.5314763709643886] 
        J_array = torch.tensor(J_array, device= device, requires_grad=True)
        P_array = torch.tensor(P_array, device= device, requires_grad=True)
        w_array = torch.tensor(w_array, device= device, requires_grad=True)

        loss_diffs = []
        prev_loss = torch.tensor(10000, device=device)
        stopping_criterion_count = 0
        file_name = f"{directory_name}/log_method_val_{time.time()}.log"
        with open(file_name, 'w') as f:
            for i in range(200):
                f.write(f"{i}\n")
                wg = WeightsGenerator(J_array, P_array, w_array, 1000, device=device, forward_mode=True)

                W = wg.generate_weight_matrix()
                tuning_curves, avg_step = executer.run_all_orientation_and_contrast(W)
                x_E, x_I = tuning_curves[:800], tuning_curves[800:]
                bessel_val = wg.validate_weight_matrix()

                f.write(f"bessel_val: {bessel_val}\n")
                trial_loss, trial_mmd_loss = loss_function.calculate_loss(x_E, y_E, x_I, y_I, avg_step, bessel_val=bessel_val)
                f.write(f"loss: {trial_loss}\n")

                trial_loss.backward()

                # GD
                J_array: torch.Tensor = (J_array - 1 * wg.J_parameters.grad).clone().detach().requires_grad_(True)
                P_array: torch.Tensor = (P_array - 1 * wg.P_parameters.grad).clone().detach().requires_grad_(True)
                w_array: torch.Tensor = (w_array - 1 * wg.w_parameters.grad).clone().detach().requires_grad_(True)

                f.write(f"{round_1D_tensor_to_list(J_array)}\n")
                f.write(f"{round_1D_tensor_to_list(P_array)}\n")
                f.write(f"{round_1D_tensor_to_list(w_array)}\n")


                loss_diffs.append(prev_loss - trial_mmd_loss.clone().detach())
                f.write(f"loss_diff: {torch.tensor(loss_diffs[-10:], device=device).mean()}\n")
                f.write("\n\n")

                if i > 40 and torch.tensor(loss_diffs[-10:], device=device).mean() < 1e-5: # This is the same stopping criterion as xNES which could be appropriate but the learning rate is different.
                    print("Early stopping")
                    if stopping_criterion_count >= 2:
                        break
                    stopping_criterion_count += 1
                else:
                    stopping_criterion_count = 0
                prev_loss = trial_mmd_loss.clone().detach()
                
                # Terminate if nan
                if torch.isnan(J_array).any().item() or torch.isnan(P_array).any().item() or torch.isnan(w_array).any().item():
                    break

                f.flush()
