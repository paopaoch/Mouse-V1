import pandas as pd
import time
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from mouse import MMDLossFunction, NeuroNN
from datetime import datetime
import sys

# Set the default data type to float32 globally
torch.set_default_dtype(torch.float32)

def training_loop(model, optimizer, Y, n=1000, device="cpu", desc="", avg_step_weighting=0.002):
    """Training loop for torch model."""

    with open(f"log_run_{time.time()}.log", "w") as f:
        # write the metadata to log file
        f.write("#### Mouse V1 Project log file ####\n\n")
        f.write(f"Code ran on the {datetime.now()}\n\n")
        f.write(f"Device: {device}\n")
        f.write(f"OS: {sys.platform}\n")
        f.write("Trainer type: With Grad\n\n")
        f.write(f"{desc}\n\n")
        f.write("Metadata:\n")
        f.write(f"Number of neurons: {model.neuron_num}\n")
        f.write(f"Number of Euler steps: {model.Nmax}\n")
        f.write(f"Record average step after {model.Navg} steps\n")
        f.write(f"Average step weighting: {avg_step_weighting}\n")
        f.write(f"Speified number of gradient descent steps: {n}\n\n")
        f.write(f"Learning rate: {optimizer.param_groups[0]['lr']}\n\n")
        f.write(f"---------------------------------------------------\n\n")
        f.write(f"Initial parameters\n")
        f.write(f"J\n")
        f.write(str(model.j_hyperparameter))
        f.write("\n")
        f.write(f"P\n")
        f.write(str(model.p_hyperparameter))
        f.write("\n")
        f.write(f"w\n")
        f.write(str(model.w_hyperparameter))
        f.write("\n\n")
        f.write(f"---------------------------------------------------\n\n")
        f.flush()

        loss_function = MMDLossFunction(device=device, avg_step_weighting=avg_step_weighting)
        model.train()

        for i in range(n):
            start = time.time()
            optimizer.zero_grad()
            preds, avg_step = model()
            end_time_simulation = time.time()
            print("Computing loss...")
            loss, MMD_loss = loss_function(preds, Y, avg_step)
            end_time_loss = time.time()
            print("Computed loss: ", loss)
            print("Backwards...")
            loss.backward()
            end_time_backward = time.time()
            optimizer.step()
            f.write(f"ITTER: {i + 1}\n")
            f.write(f"Simulation time: {end_time_simulation - start} seconds\n")
            f.write(f"Loss calculation time: {end_time_loss - end_time_simulation} seconds\n")
            f.write(f"Backwards time: {end_time_backward - end_time_loss} seconds\n")
            f.write(f"Total time taken: {time.time() - start} seconds\n")
            f.write(f"Loss: {loss}\n")
            f.write(f"avg step: {avg_step}\n")
            f.write(f"MMD loss: {MMD_loss}\n")
            f.write(f"J\n")
            f.write(str(model.j_hyperparameter))
            f.write("\n")
            f.write(f"P\n")
            f.write(str(model.p_hyperparameter))
            f.write("\n")
            f.write(f"w\n")
            f.write(str(model.w_hyperparameter))
            f.write("\n")
            f.write("\n")
            f.flush()
            print(f"DONE {i}")


def training_loop_no_backwards(model, Y, n=1000, device="cpu", desc="", avg_step_weighting=0.002):
    "Training loop for torch model."

    with open(f"log_run_{time.time()}.log", "w") as f:
        # write the metadata to log file
        f.write("#### Mouse V1 Project log file ####\n\n")
        f.write(f"Code ran on the {datetime.now()}\n\n")
        f.write(f"Device: {device}\n")
        f.write(f"OS: {sys.platform}\n")
        f.write("Trainer type: No Grad\n\n")
        f.write(f"{desc}\n\n")
        f.write("Metadata:\n")
        f.write(f"Number of neurons: {model.neuron_num}\n")
        f.write(f"Number of Euler steps: {model.Nmax}\n")
        f.write(f"Record average step after {model.Navg} steps\n")
        f.write(f"Average step weighting: {avg_step_weighting}\n")
        f.write(f"Speified number of simulated steps: {n}\n\n")
        f.write(f"---------------------------------------------------\n\n")
        f.write(f"Initial parameters\n")
        f.write(f"J\n")
        f.write(str(model.j_hyperparameter))
        f.write("\n")
        f.write(f"P\n")
        f.write(str(model.p_hyperparameter))
        f.write("\n")
        f.write(f"w\n")
        f.write(str(model.w_hyperparameter))
        f.write("\n\n")
        f.write(f"---------------------------------------------------\n\n")
        f.flush()

        loss_function = MMDLossFunction(device=device, avg_step_weighting=avg_step_weighting)

        for i in range(n):
            start = time.time()
            preds, avg_step = model()
            print("Computing loss...")
            end_time_simulation = time.time()
            loss, MMD_loss = loss_function(preds, Y, avg_step)
            end_time_loss = time.time()
            print("loss: ", loss)
            f.write(f"ITTER: {i + 1}\n")
            f.write(f"Simulation time: {end_time_simulation - start} seconds\n")
            f.write(f"Loss calculation time: {end_time_loss - end_time_simulation} seconds\n")
            f.write(f"Total time taken: {time.time() - start} seconds\n")
            f.write(f"Loss: {loss}\n")
            f.write(f"avg step: {avg_step}\n")
            f.write(f"MMD loss: {MMD_loss}\n")
            f.write(f"J\n")
            f.write(str(model.j_hyperparameter))
            f.write("\n")
            f.write(f"P\n")
            f.write(str(model.p_hyperparameter))
            f.write("\n")
            f.write(f"w\n")
            f.write(str(model.w_hyperparameter))
            f.write("\n")
            f.write("\n")
            f.flush()
            print(f"DONE {i}")



def get_data(device="cpu"):
    df = pd.read_csv("./data/K-Data.csv")
    v1 = df.query("region == 'V1'")
    m = v1.m.unique()[2]
    v1 = v1[v1.m == m] # take for all mice later
    v1 = v1.copy()  # to prevent warning
    v1["mouse_unit"] = v1["m"] + "_" + v1["u"].astype(str)
    v1 = v1.groupby(["mouse_unit", "grat_orientation", "grat_contrast", "grat_spat_freq", "grat_phase"]).mean(numeric_only=True).reset_index()
    v1 = v1[["mouse_unit", "grat_orientation", "grat_contrast", "grat_spat_freq", "grat_phase", "response"]]

    unique_units = v1['mouse_unit'].unique()
    unique_orientation = v1['grat_orientation'].unique()
    unique_contrast = v1['grat_contrast'].unique()
    unique_spat_freq = v1['grat_spat_freq'].unique()
    unique_phase = v1['grat_phase'].unique()

    shape = (len(unique_units), len(unique_orientation), len(unique_contrast), len(unique_spat_freq), len(unique_phase))
    result_array = np.full(shape, np.nan)

    # Iterate through the DataFrame and fill the array
    for _, row in tqdm(v1.iterrows()):
        u_index = np.where(unique_units == row['mouse_unit'])[0][0]
        orientation_index = np.where(unique_orientation == row['grat_orientation'])[0][0]
        contrast_index = np.where(unique_contrast == row['grat_contrast'])[0][0]
        spat_freq_index = np.where(unique_spat_freq == row['grat_spat_freq'])[0][0]
        phase_index = np.where(unique_phase == row['grat_phase'])[0][0]
        result_array[u_index, orientation_index, contrast_index, spat_freq_index, phase_index] = row['response']

    result_array = np.mean(np.mean(result_array, axis=4), axis=3)
    result_array = result_array.transpose((0, 2, 1))
    result_array = result_array * 1000

    result_array = torch.tensor(result_array, device=device)
    return result_array


if __name__ == "__main__":
    
    GRAD = False

    print("Grad: ", GRAD)
    if torch.cuda.is_available():
        device = "cuda"
        print("Model will be created on GPU")
    else:
        device = "cpu"
        print("GPU not available. Model will be created on CPU.")

    result_array = get_data(device=device)

    # J_array = [0.69, 0.64, 0., -0.29] # Max log values
    # P_array = [-2.21, -2.21, -0.8, -0.8]
    # w_array = [3.46, 3.46, 3.46, 3.46]


    J_array = [0, -0.29, 0.69, -0.64] # Keen log values
    P_array = [-0.8, -2.21, -2.21, -0.8]
    w_array = [3.46, 3.46, 3.46, 3.46]
    desc = "Starting values with Keen log values"


    if GRAD:
        model = NeuroNN(J_array, P_array, w_array, 1000, device=device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        training_loop(model, optimizer, result_array, device=device, desc=desc)
    else:
        model = NeuroNN(J_array, P_array, w_array, 1000, device=device, grad=False)
        training_loop_no_backwards(model, result_array, device=device, desc=desc)

    # https://towardsdatascience.com/how-to-use-pytorch-as-a-general-optimizer-a91cbf72a7fb
