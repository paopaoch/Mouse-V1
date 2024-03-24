import torch
from rat import MouseLossFunction, WeightsGenerator, NetworkExecuterParallel, get_data
from utils.forward_differentiator import forward_diff


def mouse_get_loss(weights_generator: WeightsGenerator, 
                network_executer: NetworkExecuterParallel, 
                loss_function: MouseLossFunction,
                y_E, y_I, feed_forward=False):
    """This function run the network and get the MMD loss"""
    if feed_forward:
        weights_FF = weights_generator.generate_feed_forward_weight_matrix()
    else:
        weights_FF = None
    weights = weights_generator.generate_weight_matrix()

    preds, avg_step = network_executer.run_all_orientation_and_contrast(weights, weights_FF)
    preds_E = preds[:weights_generator.neuron_num_e]
    preds_I = preds[weights_generator.neuron_num_e:]
    trial_loss, trial_mmd_loss = loss_function.calculate_loss(preds_E, y_E, preds_I, y_I, avg_step)
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
    loss_function = MouseLossFunction(device=device)

    trial_loss, _ = mouse_get_loss(weights_generator, network_executer, loss_function, y_E, y_I)

    return trial_loss


if __name__ == __name__:

    if torch.cuda.is_available():
        device = "cuda:1"
        print("Model will be created on GPU")
    else:
        device = "cpu"
        print("GPU not available. Model will be created on CPU.")

    params = [  torch.tensor(-5.753641449035618, device=device),
                torch.tensor(-18.152899666382492, device=device),
                torch.tensor(1.6034265007517936, device=device),
                torch.tensor(-15.163474893680885, device=device), 
                torch.tensor(-2.5418935811616112, device=device),
                torch.tensor(6.591673732008657, device=device),
                torch.tensor(-2.5418935811616112, device=device),
                torch.tensor(6.591673732008657, device=device),
                torch.tensor(-138.44395575681614, device=device),
                torch.tensor(-138.44395575681614, device=device),
                torch.tensor(-138.44395575681614, device=device),
                torch.tensor(-138.44395575681614, device=device)]

    data = get_data()

    
    lr = 1000  # Vary this learning rate according to each dimension
    for i in range(100):
        grad, loss = forward_diff(mouse_func, params, data)
        print(loss)
        for j in range(len(params)):
            params[i] = params[i] - lr * grad[i]
        print(params)
        print("---------")