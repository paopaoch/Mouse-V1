from gerbil_nes import nes_multigaussian_optim, make_torch_params
from rat import NetworkExecuterParallel, WeightsGeneratorExact
import torch

torch.manual_seed(69)

desc = "NES with JSD"

if torch.cuda.is_available():
    device = "cuda:1"
    print("Model will be created on GPU")
else:
    device = "cpu"
    print("GPU not available. Model will be created on CPU.")

# mean_list = [-1.7346010553881064, -2.586689344097943, -1.3862943611198906, -3.1780538303479458, -1.265666373331276, -0.6190392084062235, -1.265666373331276, -0.6190392084062235, -1.0986122886681098, -1.0986122886681098, -1.0986122886681098, -1.0986122886681098]
mean_list = [-2.3109, -2.8352, -1.7802, -2.2878, -2.1145, -0.4038, -1.4918, -0.1283, -1.4134, -1.4288, -0.9577, -0.6836]
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


var_list = [0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 
            0.1, 0.1, 0.1, 0.1,]

mean, cov = make_torch_params(mean_list, var_list, device=device)
nes_multigaussian_optim(mean, cov, 200, 24, y_E, y_I, device=device, neuron_num=n, desc=desc, trials=1, alpha=0.1, eta_delta=1, avg_step_weighting=0.1, stopping_criterion_step=0.0000001, adaptive_lr=False)