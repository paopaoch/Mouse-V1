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

    
    from ignore_rat import WeightsGenerator, NetworkExecuterParallel, MouseLossFunctionOptimised

    executer = NetworkExecuterParallel(1000, scaling_g=0.15)
    loss_function = MouseLossFunctionOptimised()

    # Create dataset
    J_array = [-2.059459853260332, -3.0504048076264896, -1.5877549090278045, -2.813481385641024]  # n = 1000
    P_array = [-2.0907410969337694, -0.20067069546215124, -2.0907410969337694, -0.20067069546215124]
    w_array = [-1.5314763709643886, -1.5314763709643886, -1.5314763709643886, -1.5314763709643886] 

    J_array = torch.tensor(J_array)
    P_array = torch.tensor(P_array)
    w_array = torch.tensor(w_array)
    wg = WeightsGenerator(J_array, P_array, w_array, 1000, forward_mode=True)
    W = wg.generate_weight_matrix()
    tuning_curves, avg_step = executer.run_all_orientation_and_contrast(W)
    y_E1, y_I1 = tuning_curves[:800], tuning_curves[800:]


    # Create second dataset
    J_array = [0.587113, -1.006514, -1.494353, -0.07066000000000026]
    P_array = [-3.455629, -0.830209, 0.48249700000000006, 1.6340260000000002]
    w_array = [0.6527120000000001, 1.989928, -0.7805890000000001, 0.11133499999999995]

    J_array = torch.tensor(J_array)
    P_array = torch.tensor(P_array)
    w_array = torch.tensor(w_array)
    wg = WeightsGenerator(J_array, P_array, w_array, 1000, forward_mode=True)
    W = wg.generate_weight_matrix()
    tuning_curves, avg_step = executer.run_all_orientation_and_contrast(W)
    y_E2, y_I2 = tuning_curves[:800], tuning_curves[800:]

    print(hausdorff_distance(y_E1, y_E2))
    print(hausdorff_distance(y_I1, y_I2))
