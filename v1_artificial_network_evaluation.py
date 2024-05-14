import torch
from torch import nn
from torch.utils.data import DataLoader
from v1_artificial_network import V1CNN, V1Dataset
from rat import NetworkExecuterParallel, WeightsGenerator
from utils.rodents_routine import get_device
from data_collection import trim_data
import pickle

torch.manual_seed(69)
n = 1000

device = get_device("cuda:0")
executer = NetworkExecuterParallel(n, device=device)

scale = torch.tensor([100, 100, 100, 100, 1, 1, 1, 1, 180, 180, 180, 180], device=device)

J_array = [-2.059459853260332, -3.0504048076264896, -1.5877549090278045, -2.813481385641024]  # n = 1000
P_array = [-2.0907410969337694, -0.20067069546215124, -2.0907410969337694, -0.20067069546215124]
w_array = [-1.5314763709643886, -1.5314763709643886, -1.5314763709643886, -1.5314763709643886] 

J_array = torch.tensor(J_array, device=device)
P_array = torch.tensor(P_array, device=device)
w_array = torch.tensor(w_array, device=device)
wg = WeightsGenerator(J_array, P_array, w_array, n, device=device, forward_mode=True)
W = wg.generate_weight_matrix()
tuning_curves, avg_step = executer.run_all_orientation_and_contrast(W)
y_E, y_I = tuning_curves[:800] / 99.9291, tuning_curves[800:] / 99.9940
y_E, y_I = trim_data(y_E, y_I)
y_E = torch.unsqueeze(y_E, 0)
y_I = torch.unsqueeze(y_I, 0)

PATH = "V1CNN.pth"
model = V1CNN().to(device)
model.load_state_dict(torch.load(PATH))
model.eval()

predicted = model.forward([y_E, y_I])

print(predicted * scale)

# -------------------------------

criterion = nn.MSELoss()

with open("dataset_full_for_training.pkl", 'rb') as f:
    pickle_data = pickle.load(f)

batch_size = 1
shuffle = True
test_dataset = V1Dataset(pickle_data, data_type="test", device=device)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
running_loss = 0.0
all_targets = []
all_outpus = []
for batch in test_data_loader:
    inputs = batch["input"]
    targets = batch['target']
    outputs = model.forward(inputs)
    loss = criterion(outputs, targets)
    running_loss += loss.item()

    all_targets.append(targets.clone().detach().to("cpu"))
    all_outpus.append(outputs.clone().detach().to("cpu"))

data = [all_targets, all_outpus]

with open("CNN_predicted_values.pkl", 'wb') as f:
    pickle.dump(data, f)
