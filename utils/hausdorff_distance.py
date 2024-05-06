import torch

def hausdorff_distance(set1, set2):
    distances_from_set1_to_set2 = torch.cdist(set1, set2)
    min_distances_set1_to_set2, _ = torch.min(distances_from_set1_to_set2, dim=1)
    max_dist_set1_to_set2 = torch.max(min_distances_set1_to_set2)
    
    distances_from_set2_to_set1 = torch.cdist(set2, set1)
    min_distances_set2_to_set1, _ = torch.min(distances_from_set2_to_set1, dim=1)
    max_dist_set2_to_set1 = torch.max(min_distances_set2_to_set1)
    
    hausdorff_dist = torch.max(max_dist_set1_to_set2, max_dist_set2_to_set1)
    
    return hausdorff_dist

if __name__ == "__main__":
    set1 = torch.randn((1000, 96))
    set2 = torch.randn((1000, 96))

    # Calculate Hausdorff distance using PyTorch tensors
    distance = hausdorff_distance(set1, set2)
    print("Hausdorff Distance:", distance)
