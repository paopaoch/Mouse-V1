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
        params_dict['J_EE']/40,
        params_dict['J_EI']/40,
        params_dict['J_IE']/40,
        params_dict['J_II']/40,

        params_dict['P_EE']/0.6,
        params_dict['P_EI']/0.6,
        params_dict['P_IE']/0.6,
        params_dict['P_II']/0.6,

        params_dict['w_EE']/180,
        params_dict['w_EI']/180,
        params_dict['w_IE']/180,
        params_dict['w_II']/180,
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

    if params_dict['w_EE'] < 1 or params_dict['w_EE'] > 179:
        # print(params_dict['w_EE'])
        return False
    if params_dict['w_EI'] < 1 or params_dict['w_EI'] > 179:
        # print(params_dict['w_EI'])
        return False
    if params_dict['w_IE'] < 1 or params_dict['w_IE'] > 179:
        # print(params_dict['w_IE'])
        return False
    if params_dict['w_II'] < 1 or params_dict['w_II'] > 179:
        # print(params_dict['w_II'])
        return False

    return True

data_directories = ["DATASET_bessel_large_CNN_1716323558.0135841", "DATASET_bessel_large_CNN_1716323559.0849304", "DATASET_bessel_large_CNN_1716323611.649201", "DATASET_bessel_large_CNN_1716323613.0371265", "DATASET_bessel_large_CNN_1716323614.8522983", "DATASET_bessel_large_CNN_1716323615.9924884", "DATASET_bessel_large_CNN_1716323617.280081", "DATASET_bessel_large_CNN_1716323618.9416022", "DATASET_bessel_large_CNN_1716323623.5891304", "DATASET_bessel_large_CNN_1716323632.4494338"]

# data_directories = ["DATASET_bessel_large_CNN_1716313983.985343", "DATASET_bessel_large_CNN_1716313996.184155", "DATASET_bessel_large_CNN_1716314005.419113", "DATASET_bessel_large_CNN_1716314012.959075", "DATASET_bessel_large_CNN_1716314019.666704", "DATASET_bessel_large_CNN_1716314026.5545568", "DATASET_bessel_large_CNN_1716314031.117074"]

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
        try:
            if not check_valid_range(metadata[subdir]):
                continue
        except KeyError:
            continue

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

print(len(full_dataset))

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

print("train length:", len(X_train))
print("val length:", len(X_val))
print("test length:", len(X_test))

with open("dataset_full_for_training.pkl", 'wb') as f:
    pk.dump(all_data, f)
