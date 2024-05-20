import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from utils.rodents_routine import get_device

class V1Transformer(nn.Module):
    def __init__(self, v1_model_params=13):
        super(V1Transformer, self).__init__()
        # Init positional embeddings size (104, 5)
        self.pos_embeddings = torch.randn((104, 5))  # same for both E and I?
        # cls_emb_ex = torch.randn(49)
        # cls_emb_in = torch.randn(13)

        cls_emb_ex = torch.randn(53)  # Pos embedding of size 5
        cls_emb_in = torch.randn(17)

        self.cls_emb_ex = nn.Parameter(cls_emb_ex)
        self.cls_emb_in = nn.Parameter(cls_emb_in)

        ex_encoder_layer = nn.TransformerEncoderLayer(d_model=318, nhead=6) # Input is (104, 48 + 5)
        self.ex_transformer_encoder = nn.TransformerEncoder(ex_encoder_layer, num_layers=4)

        in_encoder_layer = nn.TransformerEncoderLayer(d_model=102, nhead=6) # Input is (104, 12 + 5)
        self.in_transformer_encoder = nn.TransformerEncoder(in_encoder_layer, num_layers=4)

        self.regressor = nn.Sequential(
            nn.Linear(420,12),  # number of images * number of output channel * image width * image height
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(12, v1_model_params),  # 12 classes
            nn.Sigmoid()
        )


    def forward(self, x):  # input is a len 2 list of torch vector of size (batch_size, 104, 48 (or 12) + 1)
        batch_size = len(x[0])
        # Add pos embeddings

        pos_embeddings_E = self.pos_embeddings.unsqueeze(0).expand(x[0].size(0), x[0].size(1), self.pos_embeddings.size(1))
        ex_input = torch.cat((x[0], pos_embeddings_E), dim=2)

        pos_embeddings_I = self.pos_embeddings.unsqueeze(0).expand(x[1].size(0), x[1].size(1), self.pos_embeddings.size(1))
        in_input = torch.cat((x[1], pos_embeddings_I), dim=2)

        cls_emb_ex_holder = self.cls_emb_ex.unsqueeze(0).repeat(batch_size, 1).unsqueeze(1)
        cls_emb_in_holder = self.cls_emb_in.unsqueeze(0).repeat(batch_size, 1).unsqueeze(1)
        
        input_ex = torch.cat((cls_emb_ex_holder, ex_input), dim=1)  # -> (batch_size, 105, 49)
        input_in = torch.cat((cls_emb_in_holder, in_input), dim=1)  # -> (batch_size, 105, 13)

        input_ex = input_ex.repeat(1, 1, 6)
        input_in = input_in.repeat(1, 1, 6)

        h_ex = self.ex_transformer_encoder(input_ex)  # (batch_size, 105, 49) -> (batch_size, 105, 49)
        h_in = self.in_transformer_encoder(input_in)  # (batch_size, 105, 13) -> (batch_size, 105, 13)
        # Change this to only take the cls token embedding
        h_ex = h_ex[:, 0]  # -> (batch_size, 49)
        h_in = h_in[:, 0]  # -> (batch_size, 13)
        h_concat = torch.concatenate([h_ex, h_in], dim=1)  # -> (batch_size, 62)
        # Concatenate and regress
        # h = torch.concatenate([h_ex_avg, h_in_avg], dim=1)  # -> (batch_size, 60)
        h = self.regressor(h_concat)
        return h

class V1TransformerDataset(Dataset):
    def __init__(self, pickle_file_content, device="cpu", data_type="train"):
        self.input = pickle_file_content[f"X_{data_type}"]
        self.ground_truth_params = pickle_file_content[f"y_{data_type}"]
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
        sample["input"] = [(self.input[index][0]).to(self.device),
                           (self.input[index][1]).to(self.device)]
        return sample
    

def calc_theoretical_weights_tot(J, P, w, N_b, device="cpu"):
    """Calculate weights tot for the contraints"""
    k = 1 / (4 * w * torch.pi / 180) ** 2
    bessel: torch.Tensor = torch.special.i0(k)
    return J * torch.sqrt(torch.tensor(N_b, device=device)) * P * torch.exp(-k) * bessel


def validate_weight_matrix(model_output, device="cpu"):
    W_tot_EE = calc_theoretical_weights_tot(model_output[:,0] * 100, model_output[:,4], model_output[:,8] * 180, 800, device=device)
    W_tot_EI = calc_theoretical_weights_tot(model_output[:,1] * 100, model_output[:,5], model_output[:,9] * 180, 200, device=device)
    W_tot_IE = calc_theoretical_weights_tot(model_output[:,2] * 100, model_output[:,6], model_output[:,10] * 180, 800, device=device)
    W_tot_II = calc_theoretical_weights_tot(model_output[:,3] * 100, model_output[:,7], model_output[:,11] * 180, 200, device=device)
    
    first_condition = torch.maximum((W_tot_EE / W_tot_IE) - 1, torch.tensor(0, device=device))
    second_condition = torch.maximum((W_tot_EI / W_tot_II) - 1, torch.tensor(0, device=device))
    return torch.mean(torch.maximum(first_condition, second_condition))

if __name__ == "__main__":
    device = get_device("cuda:0")
    PATH = "V1CNN.pth"

    with open("dataset_full_for_transformer_training.pkl", 'rb') as f:
        pickle_data = pickle.load(f)

    batch_size = 8
    shuffle = True
    train_dataset = V1TransformerDataset(pickle_data, data_type="train", device=device)
    test_dataset = V1TransformerDataset(pickle_data, data_type="test", device=device)
    val_dataset = V1TransformerDataset(pickle_data, data_type="val", device=device)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = V1Transformer().to(device)
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
        running_total_loss = 0.0
        for batch in train_data_loader:
            inputs = batch["input"]
            targets = batch['target']
            
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training MSE Loss: {running_loss / len(train_data_loader):.4f}')
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Training Total Loss: {running_total_loss / len(train_data_loader):.4f}')

        # Get validation losses
        model.eval()
        running_loss = 0.0
        for batch in val_data_loader:
            inputs = batch["input"]
            targets = batch['target']
            outputs = model.forward(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Eval MSE Loss: {running_loss / len(val_data_loader):.4f}\n')
        if current_val_loss < running_loss / len(val_data_loader) and epoch > 20:
            patience -= 1
        else:
            patience = 2
            torch.save(model.state_dict(), PATH)
        
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