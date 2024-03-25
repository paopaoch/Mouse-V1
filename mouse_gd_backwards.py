import torch
import pickle
from rat import MouseLossFunctionOptimised, WeightsGenerator, NetworkExecuterParallel, get_data
from time import time

start = time()

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

J_array = [-22.907410969337693, -32.550488507104102, -17.85627263740382, -30.060150147989074]
P_array = [-3.2163953243244932, 10.833316937499324, -4.2163953243244932, 10.833316937499324]
w_array = [-135.44395575681614, -132.44395575681614, -131.44395575681614, -132.44395575681614]

J_array = torch.tensor(J_array, device= device, requires_grad=True)
P_array = torch.tensor(P_array, device= device, requires_grad=True)
w_array = torch.tensor(w_array, device= device, requires_grad=True)
executer = NetworkExecuterParallel(1000, device=device)
loss_function = MouseLossFunctionOptimised(device=device)


losses = []
for i in range(100):
    wg = WeightsGenerator(J_array, P_array, w_array, 1000, device=device, forward_mode=True)

    W = wg.generate_weight_matrix()
    tuning_curves, avg_step = executer.run_all_orientation_and_contrast(W)
    x_E, x_I = tuning_curves[:800], tuning_curves[800:]

    trial_loss, trial_mmd_loss = loss_function.calculate_loss(x_E, y_E, x_I, y_I, avg_step)

    print("loss:", trial_loss)

    trial_loss.backward()

    # GD
    J_array = (J_array - 50 * wg.J_parameters.grad).clone().detach().requires_grad_(True)
    P_array = (P_array - 5 * wg.P_parameters.grad).clone().detach().requires_grad_(True)
    w_array = (w_array - 5000 * wg.w_parameters.grad).clone().detach().requires_grad_(True)

    print(J_array)
    print(P_array)
    print(w_array)

    print("\n\n")

    losses.append(trial_loss.clone().detach())
    if i > 20 and torch.tensor(losses[-10:], device=device).mean() < 1e-5: # This is the same stopping criterion as xNES which could be appropriate but the learning rate is different.
        print("Early stopping")
        break

end = time()
print(f"Time Taken: {end - start}")