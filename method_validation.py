import torch
from mouse_nes import make_torch_params, nes_multigaussian_optim
import pickle


if __name__ == "__main__":
    desc = "Change log file log format."

    if torch.cuda.is_available():
        device = "cuda:0"
        print("Model will be created on GPU")
    else:
        device = "cpu"
        print("GPU not available. Model will be created on CPU.")

        
    mean_list = [ -8.4730, -16.5823,  13.8629, -21.9722, -1.2164, -0.0000, -4.1589,  4.1589, 40.1658, -40.1658,  81.3573,  60.5650] 
     
    var_list = [1, 1, 1, 1, 
                1, 1, 1, 1,
                1, 1, 1, 1]
    
    mean, cov = make_torch_params(mean_list, var_list, device=device)
    
    with open("data/data_1000_neurons/responses.pkl", 'rb') as f:
        responses: torch.Tensor = pickle.load(f)
        y_E, y_I = responses[:800], responses[800:]
        responses = 0
        print(y_E.shape, y_I.shape)

    print(nes_multigaussian_optim(mean, cov, 200, 24, y_E, y_I, device=device, neuron_num=1000, desc=desc, trials=1, alpha=1, eta_delta=1, avg_step_weighting=0.1))
