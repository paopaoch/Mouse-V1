import torch
import os
import random
import math

J_steep = 1
J_scale = 40

P_steep = 1
P_scale = 0.6

w_steep = 1
w_scale = 180

heter_steep = 1
heter_scale = 1

def round_1D_tensor_to_list(a, decimals=6):
    """
    Round each element of a tensor to the specified number of decimal places
    and return the result as a list of rounded values.
    
    Args:
        a (torch.Tensor): Input tensor of arbitrary length.
        decimals (int): Number of decimal places to round to.
        
    Returns:
        list: List of rounded values.
    """
    rounded_a = torch.round(a * (10 ** decimals)) / (10 ** decimals)
    rounded_list = [round(elem.item(), decimals) for elem in rounded_a]
    
    return rounded_list


def create_directory_if_not_exists(directory):
    """
    Create a directory if it does not already exist.

    Args:
        directory (str): Path of the directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
    else:
        print(f"Directory '{directory}' already exists.")


def subsample_tensor_uniform(original_tensor, x):
    """
    Subsample a tensor `original_tensor` into a smaller tensor of size [x, 8, 12]
    by uniformly spacing indices along the first dimension.

    Args:
        original_tensor (torch.Tensor): The original tensor of shape [800, 8, 12].
        x (int): Number of samples to be selected.

    Returns:
        torch.Tensor: Subsampled tensor of shape [x, 8, 12].
    """
    original_size = original_tensor.size()
    indices = torch.linspace(0, original_size[0] - 1, steps=x, dtype=torch.int64)
    subsampled_tensor = original_tensor[indices, :, :]

    return subsampled_tensor


def get_device(cuda_name="cuda:0"):
    if torch.cuda.is_available():
        device = cuda_name
        print("Model will be created on GPU")
    else:
        device = "cpu"
        print("GPU not available. Model will be created on CPU.")
    return device


def _sigmoid(value, steepness=1, scaling=1):
    return scaling / (1 + torch.exp(-steepness * value))


def _sigmoid_scalar(value, steepness=1, scaling=1):
    return scaling / (1 + math.exp(-steepness * value))


def _inverse_sigmoid(value, steepness=1, scaling=1):
    return - (1 / steepness) * torch.log((scaling / value) - 1)


J_to_params = lambda x: _inverse_sigmoid(x, J_steep, J_scale)
P_to_params = lambda x: _inverse_sigmoid(x, P_steep, P_scale)
w_to_params = lambda x: _inverse_sigmoid(x, w_steep, w_scale)


params_to_J = lambda x: _sigmoid(x, J_steep, J_scale)
params_to_P = lambda x: _sigmoid(x, P_steep, P_scale)
params_to_w = lambda x: _sigmoid(x, w_steep, w_scale)


params_to_J_scalar = lambda x: _sigmoid_scalar(x, J_steep, J_scale)
params_to_P_scalar = lambda x: _sigmoid_scalar(x, P_steep, P_scale)
params_to_w_scalar = lambda x: _sigmoid_scalar(x, w_steep, w_scale)

params_to_heter = lambda x: _sigmoid(x, heter_steep, heter_scale)
heter_to_params = lambda x: _inverse_sigmoid(x, heter_steep, heter_scale)
params_to_heter_scalar = lambda x: _sigmoid_scalar(x, heter_steep, heter_scale)
