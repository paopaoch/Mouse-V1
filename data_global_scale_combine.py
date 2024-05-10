# Get file name from the datafolder

# Select load the pickle file.

# Keep track of max value

# Store in dictionary with file name as key


from os import listdir
from os.path import join, isdir
import pickle as pk
import torch
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split


def to_normalise_tensor(params_dict):
    return torch.tensor([
        params_dict['J_EE']/100,
        params_dict['J_EI']/100,
        params_dict['J_IE']/100,
        params_dict['J_II']/100,

        params_dict['P_EE'],
        params_dict['P_EI'],
        params_dict['P_IE'],
        params_dict['P_II'],

        params_dict['w_EE']/180,
        params_dict['w_EI']/180,
        params_dict['w_IE']/180,
        params_dict['w_II']/180,
    ])


data_directories = ["DATASET_bessel_1715341433.8128526", "DATASET_bessel_1715341541.757791", "DATASET_bessel_1715341542.6103182", "DATASET_bessel_1715341593.620646", "DATASET_bessel_1715341603.251537", "DATASET_bessel_1715341607.5874019", "DATASET_bessel_1715341610.1236033", "DATASET_bessel_1715341611.6014147", "DATASET_bessel_1715341627.3461742",]

# data_directories = ["DATASET_bessel_1715379325.9177291"]

full_dataset = []
params = []
max_E = 0
max_I = 0
for directory in data_directories:
    subdirs = [f for f in listdir(directory) if isdir(join(directory, f))]
    
    df = pd.read_csv(f"{directory}/metadata.csv", dtype={'dir': str, 
                                                         'J_EE': float, 'J_EI': float, 'J_IE': float, 'J_II': float,
                                                         'P_EE': float, 'P_EI': float, 'P_IE': float, 'P_II': float,
                                                         'w_EE': float, 'w_EI': float, 'w_IE': float, 'w_II': float,}, index_col=0)
    
    metadata = df.to_dict(orient="index")

    for subdir in tqdm(subdirs):
        dataset = [None, None]
        with open(f'{directory}/{subdir}/y_E.pkl', 'rb') as f:
            data_E: torch.Tensor = pk.load(f)
            data_E = data_E.cpu()
        
        with open(f'{directory}/{subdir}/y_I.pkl', 'rb') as f:
            data_I: torch.Tensor = pk.load(f)
            data_I = data_I.cpu()
        
        
        current_data_max_E = torch.max(data_E)
        current_data_max_I = torch.max(data_I)
        
        if current_data_max_E > 100 or current_data_max_I > 100:
            continue
        
        max_E = max(max_E, current_data_max_E)
        dataset[0] = data_E

        max_I = max(max_I, current_data_max_I)
        dataset[1] = data_I

        params.append(to_normalise_tensor(metadata[subdir]))

        full_dataset.append(dataset)

print(max_E)
print(max_I)

X_train, X_test, y_train, y_test = train_test_split(full_dataset, params, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

all_data = {"max_E": max_E,
            "max_I": max_I,
            "X_train": X_train,
            "X_test": X_test,
            "X_val": X_val,
            "y_train": y_train,
            "y_test": y_test,
            "y_val": y_val,
            "params": params}

with open("dataset_full_for_training.pkl", 'wb') as f:
    pk.dump(all_data, f)