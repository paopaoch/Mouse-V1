import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from utils.rodents_routine import get_device

class V1CNN(nn.Module):
    def __init__(self, num_classes=12):
        super(V1CNN, self).__init__()
        self.features_excit = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.features_inhib = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        self.regressor = nn.Sequential(
            nn.Linear(60 * 16 * 4 * 6, 128),  # nuber of images * number of output channel * image width * image height
            nn.ReLU(),
            nn.Linear(128, 12),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(12, num_classes),  # 12 classes
            nn.Sigmoid()
        )


    def forward(self, x):  # input is a len 2 list of torch vector of size (batch_size, 48 (or 12), 8, 12)

        cnn_output_list = []
        for image in x[0].permute(1, 0, 2, 3):
            image: torch.Tensor = image.unsqueeze(1)  # image is now (batch_size, 1, 8, 12)
            image = self.features_excit(image)
            image = torch.flatten(image, 1)
            cnn_output_list.append(image)

        for image in x[1].permute(1, 0, 2, 3):
            image: torch.Tensor = image.unsqueeze(1)  # image is now (batch_size, 1, 8, 12)
            image = self.features_inhib(image)
            image = torch.flatten(image, 1)
            cnn_output_list.append(image)
        
        h = torch.concatenate(cnn_output_list, dim=1)
        h = self.regressor(h)
        
        return h


class V1Dataset(Dataset):
    def __init__(self, pickle_file_content, device="cpu", data_type="train"):
        self.input = pickle_file_content[f"X_{data_type}"]
        self.ground_truth_params = pickle_file_content[f"y_{data_type}"]
        self.max_E = pickle_file_content["max_E"]
        self.max_I = pickle_file_content["max_I"]
        self.size = len(pickle_file_content[f"X_{data_type}"])
        self.device = device

    def __len__(self):
        return self.size
    

    def __getitem__(self, index):
        sample = {
            'input': None,
            'target': None
        }

        sample["target"] = self.ground_truth_params[index].to(self.device)
        sample["input"] = [(self.input[index][0] / self.max_E).to(self.device),
                           (self.input[index][1] / self.max_I).to(self.device)]
        return sample
    
device = get_device("cuda:0")

with open("dataset_full_for_training.pkl", 'rb') as f:
    pickle_data = pickle.load(f)

batch_size = 32
shuffle = True
train_dataset = V1Dataset(pickle_data, data_type="train", device=device)
test_dataset = V1Dataset(pickle_data, data_type="test", device=device)
val_dataset = V1Dataset(pickle_data, data_type="val", device=device)

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = V1CNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
current_val_loss = 1000000
patience = 2

model.eval()
running_loss = 0.0
for batch in test_data_loader:
    inputs = batch["input"]
    targets = batch['target']
    outputs = model.forward(inputs)
    loss = criterion(outputs, targets)
    running_loss += loss.item()

print(f'\ntesting Loss: {running_loss / len(val_data_loader):.4f}\n')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in train_data_loader:
        inputs = batch["input"]
        targets = batch['target']
        
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = criterion(outputs, targets)  # need to add the bessel val and weights normalisation layer
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss / len(train_data_loader):.4f}')

    # Get validation losses
    model.eval()
    running_loss = 0.0
    for batch in val_data_loader:
        inputs = batch["input"]
        targets = batch['target']
        outputs = model.forward(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Eval Loss: {running_loss / len(val_data_loader):.4f}\n')
    if current_val_loss < running_loss / len(val_data_loader):
        patience -= 1
    else:
        patience = 2
    
    if patience == 0:
        print("Early stopping")
        break

    current_val_loss = running_loss / len(val_data_loader)

model.eval()
running_loss = 0.0
for batch in test_data_loader:
    inputs = batch["input"]
    targets = batch['target']
    outputs = model.forward(inputs)
    loss = criterion(outputs, targets)
    running_loss += loss.item()

print(f'testing Loss: {running_loss / len(val_data_loader):.4f}\n')
print(f"last test output: {outputs[-1]}")
print(f"last test target: {targets[-1]}")
print(f"last test output: {outputs[0]}")
print(f"last test target: {targets[0]}")