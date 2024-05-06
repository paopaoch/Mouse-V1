import torch
import os
import random


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