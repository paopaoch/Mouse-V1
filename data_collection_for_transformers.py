from rat import WeightsGenerator, NetworkExecuterWithSimplifiedFF
from utils.rodents_routine import get_device, create_directory_if_not_exists, round_1D_tensor_to_list, params_to_J_scalar, params_to_P_scalar, params_to_w_scalar, params_to_heter_scalar
import torch
import numpy as np
import time
import pickle
from tqdm import tqdm

if __name__ == "__main__":
    N = 1000
    E_index = 800
    device = get_device("cuda:0")

    dir_name = f"DATASET_bessel_large_{time.time()}"
    create_directory_if_not_exists(dir_name)
    metadata_file = f"{dir_name}/metadata.csv"
    with open(metadata_file, 'w') as f:
        f.write('dir,J_EE,J_EI,J_IE,J_II,P_EE,P_EI,P_IE,P_II,w_EE,w_EI,w_IE,w_II,heter_ff')


def execute_network(W, heter_ff):
    executer = NetworkExecuterWithSimplifiedFF(N, device=device)
    executer.update_heter_ff(heter_ff)
    tuning_curves, avg_step = executer.run_all_orientation_and_contrast(W)
    y_E, y_I = tuning_curves[:E_index], tuning_curves[E_index:]
    return y_E, y_I, avg_step


def get_random_param():
    J_array = torch.rand(4) * 9 - 4.5
    P_array = torch.rand(4) * 9 - 4.5
    w_array = torch.rand(4) * 9 - 4.5
    heter_ff = torch.rand(1) * 9 - 4.5
    J_array = torch.tensor(J_array, device=device, requires_grad=True)
    P_array = torch.tensor(P_array, device=device, requires_grad=True)
    w_array = torch.tensor(w_array, device=device, requires_grad=True)
    heter_ff = torch.tensor(heter_ff, device=device)
    return J_array, P_array, w_array, heter_ff


def run_gd(J_array, P_array, w_array, iterations=10, search_bessel=True):
    valid_count = 0
    found = False
    for _ in tqdm(range(iterations)):
        wg = WeightsGenerator(J_array, P_array, w_array, N, device=device, forward_mode=True)
        bessel_val = wg.validate_weight_matrix()
        loss = bessel_val
        loss.backward()
        J_array: torch.Tensor = (J_array - 1 * wg.J_parameters.grad).clone().detach().requires_grad_(True)
        P_array: torch.Tensor = (P_array - 1 * wg.P_parameters.grad).clone().detach().requires_grad_(True)
        w_array: torch.Tensor = (w_array - 1 * wg.w_parameters.grad).clone().detach().requires_grad_(True)
        
        if torch.isnan(J_array).any().item() or torch.isnan(P_array).any().item() or torch.isnan(w_array).any().item():
            found = False
            break

        if search_bessel:
            if bessel_val == 0:
                valid_count += 1
            if valid_count == 4:
                found = True
                break

    return J_array, P_array, w_array, found


def get_valid_params():
    J_array, P_array, w_array, heter_ff = get_random_param()
    J_array, P_array, w_array, found = run_gd(J_array, P_array, w_array)
    return J_array, P_array, w_array, heter_ff, found


def random_sample_from_tensor(input_tensor, x):
    total_elements = input_tensor.size(0)
    random_indices = torch.randperm(total_elements)[:x]
    sampled_tensor = input_tensor[random_indices]
    return sampled_tensor


def trim_data(y_E, y_I, size=60):
    total = len(y_E) + len(y_I)
    size_E = int((size * len(y_E) / total))
    size_I = size - size_E
    reduced_E = random_sample_from_tensor(y_E, size_E)
    reduced_I = random_sample_from_tensor(y_I, size_I)
    return reduced_E, reduced_I


def save_data(y_E, y_I, J_array, P_array, w_array, heter_ff):
    sub_dir_name = f"{time.time()}"
    full_sub_dir_name = f"{dir_name}/{sub_dir_name}"
    create_directory_if_not_exists(full_sub_dir_name)
    with open(f"{full_sub_dir_name}/y_E.pkl", "wb") as f:
        pickle.dump(y_E, f)
    with open(f"{full_sub_dir_name}/y_I.pkl", "wb") as f:
        pickle.dump(y_I, f)

    J_array = round_1D_tensor_to_list(J_array)
    P_array = round_1D_tensor_to_list(P_array)
    w_array = round_1D_tensor_to_list(w_array)

    with open(metadata_file, 'a') as f:
        f.write(f"\n{sub_dir_name}"
                + f",{round(params_to_J_scalar(J_array[0]), 4)},{round(params_to_J_scalar(J_array[1]), 4)},{round(params_to_J_scalar(J_array[2]), 4)},{round(params_to_J_scalar(J_array[3]), 4)}"
                + f",{round(params_to_P_scalar(P_array[0]), 4)},{round(params_to_P_scalar(P_array[1]), 4)},{round(params_to_P_scalar(P_array[2]), 4)},{round(params_to_P_scalar(P_array[3]), 4)}"
                + f",{round(params_to_w_scalar(w_array[0]), 4)},{round(params_to_w_scalar(w_array[1]), 4)},{round(params_to_w_scalar(w_array[2]), 4)},{round(params_to_w_scalar(w_array[3]), 4)},{round(params_to_heter_scalar(heter_ff[0]), 4)}")


def main(dataset_size=3000):
    count = 0
    i = 0
    while count <= dataset_size:
        i += 1
        
        if i > 100000:
            raise BufferError(f"Tried for too many iterations generated {count} rows")

        J_array, P_array, w_array, heter_ff, found = get_valid_params()
        J_array = J_array.requires_grad_(False)
        P_array = P_array.requires_grad_(False)
        w_array = w_array.requires_grad_(False)
        if not found:
            continue
        wg = WeightsGenerator(J_array, P_array, w_array, N, device=device)
        W = wg.generate_weight_matrix()
        y_E, y_I, _ = execute_network(W, heter_ff)
        y_E, y_I = trim_data(y_E, y_I)
        print(y_E.shape)
        save_data(y_E, y_I, J_array, P_array, w_array, heter_ff)
        count += 1
        print(count)

if __name__ == "__main__":
    main(dataset_size=10000)
