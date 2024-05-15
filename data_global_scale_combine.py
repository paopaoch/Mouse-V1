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

# data_directories = ["DATASET_bessel_1715341433.8128526", "DATASET_bessel_1715341541.757791", "DATASET_bessel_1715341542.6103182", "DATASET_bessel_1715341593.620646", "DATASET_bessel_1715341603.251537", "DATASET_bessel_1715341607.5874019", "DATASET_bessel_1715341610.1236033", "DATASET_bessel_1715341611.6014147", "DATASET_bessel_1715341627.3461742", "DATASET_bessel_1715390826.6967611", "DATASET_bessel_1715390828.301663", "DATASET_bessel_1715390830.7035437", "DATASET_bessel_1715390845.0148275", "DATASET_bessel_1715390848.9291334", "DATASET_bessel_1715390850.7687511", "DATASET_bessel_1715390870.609546", "DATASET_bessel_1715390874.2647386", "DATASET_bessel_1715477976.4489124", "DATASET_bessel_1715477978.126739", "DATASET_bessel_1715477980.562683", "DATASET_bessel_1715477982.479733", "DATASET_bessel_1715477984.690376", "DATASET_bessel_1715477985.8544126", "DATASET_bessel_1715478002.6127565"]
data_directories =  ["DATASET_bessel_large_1715784155.5120544", "DATASET_bessel_large_1715784155.585425", "DATASET_bessel_large_1715784157.623016", "DATASET_bessel_large_1715784159.9759786", "DATASET_bessel_large_1715784162.2602618", "DATASET_bessel_large_1715784163.6065583", "DATASET_bessel_large_1715784191.6599987", "DATASET_bessel_large_1715784194.9939306", "DATASET_bessel_large_1715784224.8111737", "DATASET_bessel_large_1715784226.5055614", "DATASET_bessel_large_1715784229.0195346", "DATASET_bessel_large_1715784229.3999553", "DATASET_bessel_large_1715784245.415284", "DATASET_bessel_large_1715784247.3760035", "DATASET_bessel_large_1715784254.8295689", "DATASET_bessel_large_1715784264.0894055"]
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
