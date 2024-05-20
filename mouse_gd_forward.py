import torch
from rat import MouseLossFunctionHomogeneous, WeightsGenerator, NetworkExecuterWithSimplifiedFF, get_data
from utils.forward_differentiator import forward_diff

import time
from datetime import datetime
import sys
import socket
import pickle

from utils.rodents_routine import get_device, create_directory_if_not_exists


def mouse_get_loss(weights_generator: WeightsGenerator, 
                network_executer: NetworkExecuterWithSimplifiedFF, 
                loss_function: MouseLossFunctionHomogeneous,
                y_E, y_I, feed_forward=False, heter_ff=None):
    """This function run the network and get the MMD loss"""
    if feed_forward:
        weights_FF = weights_generator.generate_feed_forward_weight_matrix()
    else:
        weights_FF = None
    weights = weights_generator.generate_weight_matrix()
    bessel_val = weights_generator.validate_weight_matrix()
    network_executer.update_heter_ff(heter_ff)
    preds, avg_step = network_executer.run_all_orientation_and_contrast(weights, weights_FF)
    preds_E = preds[:weights_generator.neuron_num_e]
    preds_I = preds[weights_generator.neuron_num_e:]
    trial_loss, trial_mmd_loss = loss_function.calculate_loss(preds_E, y_E, preds_I, y_I, avg_step, bessel_val=bessel_val)
    return trial_loss, trial_mmd_loss


def extract_params(params):
    J_array = [params[0], params[1], params[2], params[3]]
    P_array = [params[4], params[5], params[6], params[7]]
    w_array = [params[8], params[9], params[10], params[11]]
    heter_ff = params[12]
    return J_array, P_array, w_array, heter_ff


def mouse_func(params: list, hyperparams: list, device="cpu"):

    J_array, P_array, w_array, heter_ff = extract_params(params)
    y_E, y_I = hyperparams
    weights_generator = WeightsGenerator(J_array, P_array, w_array, 10000, forward_mode=True, device=device)  # TODO: Change this to variable
    network_executer = NetworkExecuterWithSimplifiedFF(10000, device=device)
    loss_function = MouseLossFunctionHomogeneous(device=device)

    trial_loss, _ = mouse_get_loss(weights_generator, network_executer, loss_function, y_E, y_I, heter_ff=heter_ff)

    return trial_loss


def draw_new_parameters(device):
    random_tensors = []
    for _ in range(13):
        tensor = torch.rand(1, device=device) * 9 - 4.5
        random_tensors.append(tensor[0])
    
    return random_tensors


def params_to_list(params):
    return [float(val) for val in params]

if __name__ == __name__:
    start = time.time()

    device = get_device("cuda:1")
    # torch.manual_seed(69)

    dir_name = f"method_validation_forward_{time.time()}"
    create_directory_if_not_exists(dir_name)

    # Create dataset
    J_array = [-0.9308613398652443, -2.0604571635972393, -0.30535063458645906, -1.802886963254238]
    P_array = [-1.493925025312256, 1.09861228866811, -1.493925025312256, 1.09861228866811]
    w_array = [-1.5314763709643886, -1.5314763709643886, -1.5314763709643886, -1.5314763709643886] 
    heter_ff = torch.tensor([-1.3862943611198906], device=device)

    executer = NetworkExecuterWithSimplifiedFF(10000, device=device)
    J_array = torch.tensor(J_array, device=device)
    P_array = torch.tensor(P_array, device=device)
    w_array = torch.tensor(w_array, device=device)
    wg = WeightsGenerator(J_array, P_array, w_array, 10000, device=device, forward_mode=True)
    W = wg.generate_weight_matrix()
    executer.update_heter_ff(heter_ff)
    tuning_curves, avg_step = executer.run_all_orientation_and_contrast(W)
    y_E, y_I = tuning_curves[:8000], tuning_curves[8000:]
    
    lr = [1 for _ in range(13)]

    for _ in range(30):
        params = draw_new_parameters(device=device)
        
        file_name = f"log_forward_diff_{time.time()}.log"

        with open(f"{dir_name}/{file_name}", "w") as f:
            f.write("#####Forward mode mouse v1 log file#####\n\n\n")
            f.write(f"Code ran on the {datetime.now()}\n\n")
            f.write(f"Run from xNES minima\n")
            f.write(f"Device: {device}\n")
            f.write(f"OS: {sys.platform}\n")
            f.write(f"Machine: {socket.gethostname()}\n")
            f.write(f"\n\n")
            f.write("Metadata:\n")
            f.write(f"learning rates: {lr}\n\n")
            f.write(f"initial params: {params}\n\n")
            f.write("----------------------------\n")
            f.flush()

            loss_diffs = []
            prev_loss = torch.tensor(10000, device=device)
            stopping_criterion_count = 0

            for i in range(200):
                grad, loss = forward_diff(mouse_func, params, (y_E, y_I), device=device)
                print(loss)
                for j in range(len(params)):
                    params[j] = params[j] - lr[j] * grad[j]
                f.write(f"{i}\n")
                f.write(f"loss: {float(loss)}\n")
                f.write(f"params: {params_to_list(params)}\n")
                f.write("----------------------------\n\n\n")
                f.flush()

                for item in params:
                    if torch.isnan(item).any():
                        break_loop = True
                        break
                if break_loop:
                    break

                loss_diffs.append(prev_loss - loss.clone().detach())
                print(torch.tensor(loss_diffs[-10:], device=device).mean())
                if i > 30 and torch.tensor(loss_diffs[-10:], device=device).mean() < 1e-6: # This is the same stopping criterion as xNES which could be appropriate but the learning rate is different.
                    f.write("Early stopping\n")
                    if stopping_criterion_count >= 2:
                        break
                    stopping_criterion_count += 1
                else:
                    stopping_criterion_count = 0
                prev_loss = loss.clone().detach()


            end = time.time()
            f.write(f"time taken: {end - start}\n")
            f.write(f"number of iterations: {i + 1}\n")
            f.write(f"average time per iter: {(end - start) / (i + 1)}\n")
            f.flush()