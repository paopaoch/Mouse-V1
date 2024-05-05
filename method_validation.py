import torch
from rat import WeightsGenerator, NetworkExecuterParallel, MouseLossFunctionOptimised


if __name__ == "__main__":
    desc = "Validate stopping criterion"

    if torch.cuda.is_available():
        device = "cuda:1"
        print("Model will be created on GPU")
    else:
        device = "cpu"
        print("GPU not available. Model will be created on CPU.")

    executer = NetworkExecuterParallel(1000, device=device)
    loss_function = MouseLossFunctionOptimised(device=device)


    # Create dataset
    J_array = [-2.059459853260332, -3.0504048076264896, -1.5877549090278045, -2.813481385641024]  # n = 1000
    P_array = [-2.0907410969337694, -0.20067069546215124, -2.0907410969337694, -0.20067069546215124]
    w_array = [-1.5314763709643886, -1.5314763709643886, -1.5314763709643886, -1.5314763709643886] 

    J_array = torch.tensor(J_array, device= device)
    P_array = torch.tensor(P_array, device= device)
    w_array = torch.tensor(w_array, device= device)
    wg = WeightsGenerator(J_array, P_array, w_array, 1000, device=device, forward_mode=True)
    W = wg.generate_weight_matrix()
    tuning_curves, avg_step = executer.run_all_orientation_and_contrast(W)
    y_E, y_I = tuning_curves[:800], tuning_curves[800:]


    J_array = [-2.059459853260332, -3.0504048076264896, -1.5877549090278045, -2.813481385641024]  # n = 1000
    P_array = [-2.0907410969337694, -0.20067069546215124, -2.0907410969337694, -0.20067069546215124]
    w_array = [-1.5314763709643886, -1.5314763709643886, -1.5314763709643886, -1.5314763709643886] 
    J_array = torch.tensor(J_array, device= device, requires_grad=True)
    P_array = torch.tensor(P_array, device= device, requires_grad=True)
    w_array = torch.tensor(w_array, device= device, requires_grad=True)

    loss_diffs = []
    prev_loss = torch.tensor(10000, device=device)
    stopping_criterion_count = 0
    for i in range(200):
        print(i)
        wg = WeightsGenerator(J_array, P_array, w_array, 1000, device=device, forward_mode=True)

        W = wg.generate_weight_matrix()
        tuning_curves, avg_step = executer.run_all_orientation_and_contrast(W)
        x_E, x_I = tuning_curves[:800], tuning_curves[800:]
        bessel_val = wg.validate_weight_matrix()

        print("bessel_val:", bessel_val)
        trial_loss, trial_mmd_loss = loss_function.calculate_loss(x_E, y_E, x_I, y_I, avg_step, bessel_val=bessel_val)
        print("loss:", float(trial_loss))

        trial_loss.backward()

        # GD
        J_array = (J_array - 1 * wg.J_parameters.grad).clone().detach().requires_grad_(True)
        P_array = (P_array - 1 * wg.P_parameters.grad).clone().detach().requires_grad_(True)
        w_array = (w_array - 1 * wg.w_parameters.grad).clone().detach().requires_grad_(True)

        print(J_array)
        print(P_array)
        print(w_array)


        loss_diffs.append(prev_loss - trial_mmd_loss.clone().detach())
        print("loss_diff", torch.tensor(loss_diffs[-10:], device=device).mean())
        print("\n\n")

        if i > 40 and torch.tensor(loss_diffs[-10:], device=device).mean() < 1e-5: # This is the same stopping criterion as xNES which could be appropriate but the learning rate is different.
            print("Early stopping")
            if stopping_criterion_count >= 2:
                break
            stopping_criterion_count += 1
        else:
            stopping_criterion_count = 0
        prev_loss = trial_mmd_loss.clone().detach()