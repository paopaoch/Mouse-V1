import torch
import pickle
from rat import MouseLossFunctionOptimised, WeightsGenerator, NetworkExecuterParallel, get_data

if torch.cuda.is_available():
    device = "cuda:0"
    print("Model will be created on GPU")
else:
    device = "cpu"
    print("GPU not available. Model will be created on CPU.")

# with open("./data/data_1000_neuron3/responses.pkl", 'rb') as f:
with open("/Users/paopao_ch/Documents/projects/v1_modelling/Mouse-V1-Pytorch/plots/ignore_plots_1710350762.007822/responses.pkl", 'rb') as f:
    responses: torch.Tensor = pickle.load(f)
    responses = responses.to(device)
    y_E, y_I = responses[:8000], responses[8000:]
    responses = 0

J_array = [-20.907410969337693, -30.550488507104102, -15.85627263740382, -28.060150147989074]
P_array = [-1.2163953243244932, 8.833316937499324, -1.2163953243244932, 8.833316937499324]
w_array = [-138.44395575681614, -138.44395575681614, -138.44395575681614, -138.44395575681614]

wg = WeightsGenerator(J_array, P_array, w_array, 10000, requires_grad=True, device=device)
executer = NetworkExecuterParallel(10000, device=device)

W = wg.generate_weight_matrix()
tuning_curves, avg_step = executer.run_all_orientation_and_contrast(W)
x_E, x_I = tuning_curves[:8000], tuning_curves[8000:]

loss_function = MouseLossFunctionOptimised(device=device)
trial_loss, trial_mmd_loss = loss_function.calculate_loss(x_E, y_E, x_I, y_I, avg_step)

print(trial_loss)

trial_loss.backward()

print(wg.J_parameters.grad)
