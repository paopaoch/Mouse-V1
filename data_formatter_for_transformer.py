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
from rat import MouseLossFunctionHomogeneous


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

        params_dict['heter_ff'],
    ])


def check_valid_range(params_dict):
    if params_dict['J_EE'] < 1 and params_dict['J_EE'] > 90:
        # print(params_dict['J_EE'])
        return False
    if params_dict['J_EI'] < 1 and params_dict['J_EE'] > 90:
        # print(params_dict['J_EI'])
        return False
    if params_dict['J_IE'] < 1 and params_dict['J_EE'] > 90:
        # print(params_dict['J_IE'])
        return False
    if params_dict['J_II'] < 1 and params_dict['J_EE'] > 90:
        # print(params_dict['J_II'])
        return False

    if params_dict['P_EE'] < 0.001:
        # print(params_dict['P_EE'])
        return False
    if params_dict['P_EI'] < 0.001:
        # print(params_dict['P_EI'])
        return False
    if params_dict['P_IE'] < 0.001:
        # print(params_dict['P_IE'])
        return False
    if params_dict['P_II'] < 0.001:
        # print(params_dict['P_II'])
        return False

    if params_dict['w_EE'] < 5 or params_dict['w_EE'] > 175:
        # print(params_dict['w_EE'])
        return False
    if params_dict['w_EI'] < 5 or params_dict['w_EI'] > 175:
        # print(params_dict['w_EI'])
        return False
    if params_dict['w_IE'] < 5 or params_dict['w_IE'] > 175:
        # print(params_dict['w_IE'])
        return False
    if params_dict['w_II'] < 5 or params_dict['w_II'] > 175:
        # print(params_dict['w_II'])
        return False

    return True

data_directories = [ "DATASET_bessel_large_1716244462.946338"
, "DATASET_bessel_large_1716244464.3948903"
, "DATASET_bessel_large_1716244467.3820353"
, "DATASET_bessel_large_1716244468.8485663"
, "DATASET_bessel_large_1716244469.969146"
, "DATASET_bessel_large_1716244470.8195784"
, "DATASET_bessel_large_1716244471.4252684"
, "DATASET_bessel_large_1716244471.905603"
, "DATASET_bessel_large_1716244472.636885"
]
# data_directories = ["DATASET_bessel_large_1716224919.619017", "DATASET_bessel_large_1716244969.294642", "DATASET_bessel_large_1716244993.482489", "DATASET_bessel_large_1716244993.482489", "DATASET_bessel_large_1716245868.050762", "DATASET_bessel_large_1716245873.863752"]

loss_function = MouseLossFunctionHomogeneous()

full_dataset = []
params = []
max_E = 100
max_I = 100
for directory in data_directories:
    subdirs = [f for f in listdir(directory) if isdir(join(directory, f))]
    
    df = pd.read_csv(f"{directory}/metadata.csv", dtype={'dir': str, 
                                                         'J_EE': float, 'J_EI': float, 'J_IE': float, 'J_II': float,
                                                         'P_EE': float, 'P_EI': float, 'P_IE': float, 'P_II': float,
                                                         'w_EE': float, 'w_EI': float, 'w_IE': float, 'w_II': float,
                                                         'heter_ff': float}, index_col=0)
    
    metadata = df.to_dict(orient="index")

    for subdir in tqdm(subdirs):
        try:
            if not check_valid_range(metadata[subdir]):
                continue
        except KeyError:
            continue

        dataset = [None, None]
        with open(f'{directory}/{subdir}/y_E.pkl', 'rb') as f:
            data_E: torch.Tensor = pk.load(f)
            data_E = data_E / max_E
            data_E = data_E.cpu()
        
        with open(f'{directory}/{subdir}/y_I.pkl', 'rb') as f:
            data_I: torch.Tensor = pk.load(f)
            data_I = data_I / max_I
            data_I = data_I.cpu()

        # Centralise Curves
        data_E = loss_function.centralise_all_curves(data_E)
        data_I = loss_function.centralise_all_curves(data_I)

        # Homogeneous Curves
        data_E, contrast_vector_E = loss_function._transform_all_tuning_curves(data_E)
        data_I, contrast_vector_I = loss_function._transform_all_tuning_curves(data_I)
        
        # flatten and combine
        data_E = data_E.flatten(start_dim=1)
        data_E = torch.concatenate([data_E, contrast_vector_E], dim=1).transpose(1, 0)
        data_I = data_I.flatten(start_dim=1)
        data_I = torch.concatenate([data_I, contrast_vector_I], dim=1).transpose(1, 0)
        
        if torch.max(data_E) > 100 or torch.max(data_I) > 100:
            continue

        dataset[0] = data_E
        dataset[1] = data_I

        params.append(to_normalise_tensor(metadata[subdir]))

        full_dataset.append(dataset)

print(len(full_dataset))

X_train, X_test, y_train, y_test = train_test_split(full_dataset, params, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

all_data = {"X_train": X_train,
            "X_test": X_test,
            "X_val": X_val,
            "y_train": y_train,
            "y_test": y_test,
            "y_val": y_val,
            "params": params}

print("train length:", len(X_train))
print("val length:", len(X_val))
print("test length:", len(X_test))

with open("dataset_full_for_transformer_training.pkl", 'wb') as f:
    pk.dump(all_data, f)
