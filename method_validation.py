import torch
from mouse_nes import make_torch_params, nes_multigaussian_optim
import pickle


if __name__ == "__main__":
    desc = "Removing important mixing seems to work but we will try to increase the learning rate as well as the covariance matrix size for a faster convergence."

    if torch.cuda.is_available():
        device = "cuda:0"
        print("Model will be created on GPU")
    else:
        device = "cpu"
        print("GPU not available. Model will be created on CPU.")

        
    mean_list = [-8.472978603872036, -16.582280766035325, 13.862943611198906, -21.972245773362197, -1.2163953243244932, -0.0, -4.1588830833596715, 4.1588830833596715, 40.165839236557744, -40.16583923655776, 81.35732227375028, 60.56500259181835] 
     
    var_list = [0.1, 0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1, 
                0.1, 0.1, 0.1, 0.1]
    
    mean, cov = make_torch_params(mean_list, var_list, device=device)
    
    with open("data/data_1000_neurons/responses.pkl", 'rb') as f:
        responses: torch.Tensor = pickle.load(f)
        y_E, y_I = responses[:800], responses[800:]
        responses = 0
        print(y_E.shape, y_I.shape)

    print(nes_multigaussian_optim(mean, cov, 200, 12, y_E, y_I, device=device, neuron_num=1000, desc=desc, trials=3, alpha=1, eta_delta=1, avg_step_weighting=0.0001))
