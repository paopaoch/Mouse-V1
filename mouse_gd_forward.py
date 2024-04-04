import torch
from rat import MouseLossFunctionOptimised, WeightsGenerator, NetworkExecuterParallel, get_data
from utils.forward_differentiator import forward_diff

import time
from datetime import datetime
import sys
import socket
import pickle


def mouse_get_loss(weights_generator: WeightsGenerator, 
                network_executer: NetworkExecuterParallel, 
                loss_function: MouseLossFunctionOptimised,
                y_E, y_I, feed_forward=False):
    """This function run the network and get the MMD loss"""
    if feed_forward:
        weights_FF = weights_generator.generate_feed_forward_weight_matrix()
    else:
        weights_FF = None
    weights = weights_generator.generate_weight_matrix()
    bessel_val = weights_generator.validate_weight_matrix()

    preds, avg_step = network_executer.run_all_orientation_and_contrast(weights, weights_FF)
    preds_E = preds[:weights_generator.neuron_num_e]
    preds_I = preds[weights_generator.neuron_num_e:]
    trial_loss, trial_mmd_loss = loss_function.calculate_loss(preds_E, y_E, preds_I, y_I, avg_step, bessel_val=bessel_val)
    return trial_loss, trial_mmd_loss


def extract_params(params):
    J_array = [params[0], params[1], params[2], params[3]]
    P_array = [params[4], params[5], params[6], params[7]]
    w_array = [params[8], params[9], params[10], params[11]]
    return J_array, P_array, w_array


def mouse_func(params: list, hyperparams: list, device="cpu"):

    J_array, P_array, w_array = extract_params(params)
    y_E, y_I = hyperparams
    weights_generator = WeightsGenerator(J_array, P_array, w_array, 1000, forward_mode=True, device=device)
    network_executer = NetworkExecuterParallel(1000, device=device)
    loss_function = MouseLossFunctionOptimised(device=device)

    trial_loss, _ = mouse_get_loss(weights_generator, network_executer, loss_function, y_E, y_I)

    return trial_loss


if __name__ == __name__:

    start = time.time()

    if torch.cuda.is_available():
        device = "cuda:1"
        print("Model will be created on GPU")
    else:
        device = "cpu"
        print("GPU not available. Model will be created on CPU.")

    # params = [torch.tensor(-5.753641449035618, device=device),
    #           torch.tensor(-18.152899666382492, device=device),
    #           torch.tensor(1.6034265007517936, device=device),
    #           torch.tensor(-15.163474893680885, device=device),
    #           torch.tensor(-2.5418935811616112, device=device),
    #           torch.tensor(6.591673732008657, device=device),
    #           torch.tensor(-2.5418935811616112, device=device),
    #           torch.tensor(6.591673732008657, device=device),
    #           torch.tensor(-138.44395575681614, device=device),
    #           torch.tensor(-138.44395575681614, device=device),
    #           torch.tensor(-138.44395575681614, device=device),
    #           torch.tensor(-138.44395575681614, device=device)]
        
    params = [torch.tensor(-22.907410969337693, device=device),
               torch.tensor(-32.550488507104102, device=device),
               torch.tensor(-17.85627263740382, device=device),
               torch.tensor(-30.060150147989074, device=device),
               torch.tensor(-3.2163953243244932, device=device),
               torch.tensor(10.833316937499324, device=device),
               torch.tensor(-4.2163953243244932, device=device),
               torch.tensor(10.833316937499324, device=device),
               torch.tensor(-135.44395575681614, device=device),
               torch.tensor(-132.44395575681614, device=device),
               torch.tensor(-131.44395575681614, device=device),
               torch.tensor(-132.44395575681614, device=device)]  # use this for 1000
        
    # params = [torch.tensor(-5., device=device),
    #           torch.tensor(-18., device=device),
    #           torch.tensor(2., device=device),
    #           torch.tensor(-15., device=device),
    #           torch.tensor(-2., device=device),
    #           torch.tensor(6., device=device),
    #           torch.tensor(-2., device=device),
    #           torch.tensor(6., device=device),
    #           torch.tensor(-135., device=device),
    #           torch.tensor(-132., device=device),
    #           torch.tensor(-131., device=device),
    #           torch.tensor(-132., device=device)]  # Use this for 10000
        
    # params = [torch.tensor(-6.5124, device=device),
    #           torch.tensor(-18.3093, device=device),
    #           torch.tensor(1.9300, device=device),
    #           torch.tensor(-15.0095, device=device),
    #           torch.tensor(-2.5466, device=device),
    #           torch.tensor(5.1354, device=device),
    #           torch.tensor(-2.6696, device=device),
    #           torch.tensor(4.7931, device=device),
    #           torch.tensor(-146.3552, device=device),
    #           torch.tensor(-141.0944, device=device),
    #           torch.tensor(-150.1857, device=device),
    #           torch.tensor(-124.6875, device=device)]  # From xNES


    # data = get_data(device=device)

    with open("./data/data_1000_neuron3/responses.pkl", 'rb') as f:
        responses: torch.Tensor = pickle.load(f)
        responses = responses.to(device)
        data = (responses[:800], responses[800:])
        responses = 0

    # with open("./plots/ignore_plots_config13/responses.pkl", 'rb') as f:
    #     responses: torch.Tensor = pickle.load(f)
    #     responses = responses.to(device)
    #     data = responses[:8000], responses[8000:]
    #     responses = 0
    
    # lr = 1000  # Vary this learning rate according to each dimension
    # lr = [10000, 10000, 10000, 10000,
    #       1000, 1000, 1000, 1000,
    #       1000000, 1000000, 1000000, 1000000,]  # Seems to work well will actual data
        
    lr = [50, 50, 50, 50,
          5, 5, 5, 5,
          5000, 5000, 5000, 5000,]  # Seems to work well with simulated data
    
    # lr = [5, 5, 5, 5,
    #       .5, .5, .5, .5,
    #       500, 500, 500, 500,]  # ten times less to minimise from xNES minima
    
    file_name = f"log_forward_diff_{time.time()}.log"

    with open(file_name, "w") as f:
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

        for i in range(200):  # need to implement a stopping criterion
            grad, loss = forward_diff(mouse_func, params, data, device=device)
            print(loss)
            for j in range(len(params)):
                params[j] = params[j] - lr[j] * grad[j]
            f.write(f"{i}\n")
            f.write(f"loss: {float(loss)}\n")
            f.write(f"params: {params}\n")
            f.write("----------------------------\n\n\n")
            f.flush()

            loss_diffs.append(prev_loss - loss.clone().detach())
            print(torch.tensor(loss_diffs[-10:], device=device).mean())
            if i > 30 and torch.tensor(loss_diffs[-10:], device=device).mean() < 1e-5: # This is the same stopping criterion as xNES which could be appropriate but the learning rate is different.
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