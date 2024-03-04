import torch
from mouse_nes import make_torch_params, nes_multigaussian_optim
import pickle


if __name__ == "__main__":
    desc = "Validate stopping criterion"

    if torch.cuda.is_available():
        device = "cuda:0"
        print("Model will be created on GPU")
    else:
        device = "cpu"
        print("GPU not available. Model will be created on CPU.")

        
    # mean_list = [ -8.4730, -16.5823,  13.8629, -21.9722, -1.2164, -0.0000, -4.1589,  4.1589, 40.1658, -40.1658,  81.3573,  60.5650] 
    # mean_list = [-4.054651081081644, -17.346010553881065, 8.472978603872036, -15.85627263740382, -30.346010553881065, -30.346010553881065, -10.990684938388938, -1.2163953243244932, -8.83331693749932, -1.2163953243244932, 5.2163953243244932, 5.2163953243244932, -255.84942256760897, -304.50168192079303, -214.12513203729057, -255.84942256760897, -350.50168192079303, -350.50168192079303]
    mean_list = [-4.054651081081644, -19.924301646902062, -0.0, -12.083112059245341, -6.591673732008658, 1.8571176252186712, -4.1588830833596715, 4.549042468104266, -167.03761889472233, -187.23627477210516, -143.08737747657977, -167.03761889472233]
    var_list = [1, 1, 1, 1, 
                1, 1, 1, 1,
                1, 1, 1, 1,]
    
    mean, cov = make_torch_params(mean_list, var_list, device=device)
    
    with open("data/data_1000_neurons2/responses.pkl", 'rb') as f:
        responses: torch.Tensor = pickle.load(f)
        y_E, y_I = responses[:800], responses[800:]
        responses = 0
        print(y_E.shape, y_I.shape)

    print(nes_multigaussian_optim(mean, cov, 100, 12, y_E, y_I, device=device, neuron_num=1000, desc=desc, trials=1, alpha=0.1, eta_delta=1, avg_step_weighting=0.1, min_iter=15, stopping_criterion_step=0.001, stopping_criterion_tolerance=2))
