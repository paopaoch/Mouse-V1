import torch
from mouse_nes import make_torch_params, nes_multigaussian_optim
from rat import get_data

if __name__ == "__main__":
    desc = "For finding lambda"

    if torch.cuda.is_available():
        device = "cuda:0"
        print("Model will be created on GPU")
    else:
        device = "cpu"
        print("GPU not available. Model will be created on CPU.")

    mean_list = [-5.753641449035618, -18.152899666382492, 1.6034265007517936, -15.163474893680885, -2.5418935811616112, 6.591673732008657, -2.5418935811616112, 6.591673732008657, -138.44395575681614, -138.44395575681614, -138.44395575681614, -138.44395575681614]  # Config 13
     
    # var_list = [5, 5, 5, 5, 
    #             1, 1, 1, 1, 
    #             250, 250, 250, 250]  # This is from the experiment with 1000 neurons and with simulated data

    var_list = [2.5, 2.5, 2.5, 2.5, 
                0.5, 0.5, 0.5, 0.5, 
                125, 125, 125, 125]  # This is from the experiment with 1000 neurons and with simulated data
    
    mean, cov = make_torch_params(mean_list, var_list, device=device)

    y_E, y_I = get_data(device=device)
    
    lambdas = [12, 66]
    # lambdas = [18, 60]
    # lambdas = [24, 36, 54]
    # lambdas = [30, 42, 48]
    
    for lambda_ in lambdas:
        for _ in range(4):
            nes_multigaussian_optim(mean, cov, lambda_, 48, y_E, y_I, device=device, neuron_num=10000, desc=desc, trials=1, alpha=0.1, eta_delta=1, avg_step_weighting=0.1, stopping_criterion_step=0.000005)
