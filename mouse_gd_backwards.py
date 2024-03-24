import torch
import pickle
from rat import MouseLossFunctionOptimised, WeightsGenerator, NetworkExecuterParallel, get_data

if torch.cuda.is_available():
    device = "cuda:0"
    print("Model will be created on GPU")
else:
    device = "cpu"
    print("GPU not available. Model will be created on CPU.")

with open("./data/data_1000_neuron3/responses.pkl", 'rb') as f:
# with open("/Users/paopao_ch/Documents/projects/v1_modelling/Mouse-V1-Pytorch/plots/ignore_plots_1710350762.007822/responses.pkl", 'rb') as f:
    responses: torch.Tensor = pickle.load(f)
    responses = responses.to(device)
    y_E, y_I = responses[:800], responses[800:]
    responses = 0

# J_array = [-20.907410969337693, -30.550488507104102, -15.85627263740382, -28.060150147989074]
# P_array = [-1.2163953243244932, 8.833316937499324, -1.2163953243244932, 8.833316937499324]
# w_array = [-138.44395575681614, -138.44395575681614, -138.44395575681614, -138.44395575681614]
    
J_array = torch.tensor([-22.907410969337693, -32.550488507104102, -17.85627263740382, -30.060150147989074], device= device, requires_grad=True)
P_array = torch.tensor([-3.2163953243244932, 10.833316937499324, -4.2163953243244932, 10.833316937499324], device= device, requires_grad=True)
w_array = torch.tensor([-135.44395575681614, -132.44395575681614, -131.44395575681614, -132.44395575681614], device= device, requires_grad=True)

for i in range(100):
    wg = WeightsGenerator(J_array, P_array, w_array, 1000, device=device, forward_mode=True)
    executer = NetworkExecuterParallel(1000, device=device)

    W = wg.generate_weight_matrix()
    tuning_curves, avg_step = executer.run_all_orientation_and_contrast(W)
    x_E, x_I = tuning_curves[:800], tuning_curves[800:]

    loss_function = MouseLossFunctionOptimised(device=device)
    trial_loss, trial_mmd_loss = loss_function.calculate_loss(x_E, y_E, x_I, y_I, avg_step)

    print(trial_loss)

    trial_loss.backward()

    # GD
    J_array = J_array - 10000 * wg.J_parameters.grad
    P_array = P_array - 1000 * wg.P_parameters.grad
    w_array = w_array - 1000000 * wg.w_parameters.grad
