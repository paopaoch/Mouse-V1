import torch
from rat import WeightsGenerator, NetworkExecuterWithSimplifiedFF, MouseLossFunctionHomogeneous
import time
from utils.rodents_routine import get_device
import pickle

torch.manual_seed(69)

if __name__ == "__main__":

    directory_name = f"method_val_log_{time.time()}"
    # create_directory_if_not_exists(directory_name)

    desc = "Method validation multi dataset"

    device = get_device("cuda:1")
    
    restart_num = 40
    dataset_trials = 30

    executer = NetworkExecuterWithSimplifiedFF(1000, device=device, scaling_g=0.15)
    loss_function = MouseLossFunctionHomogeneous(device=device)

    # Config 13
    config_J_array = torch.tensor([-0.9308613398652443, -2.0604571635972393, -0.30535063458645906, -1.802886963254238], device=device)  # n = 1000
    config_P_array = torch.tensor([-1.493925025312256, 1.09861228866811, -1.493925025312256, 1.09861228866811], device=device)
    config_w_array = torch.tensor([-1.5314763709643886, -1.5314763709643886, -1.5314763709643886, -1.5314763709643886], device=device)
    config_heter_ff = torch.tensor([0.2], device=device)

    # Create a lower and upper bound
    lower_J = config_J_array * 0.8
    upper_J = config_J_array * 1.2
    lower_P = config_P_array * 0.8
    upper_P = config_P_array * 1.2
    lower_w = config_w_array * 0.8
    upper_w = config_w_array * 1.2
    lower_heter_ff = config_heter_ff * 0.8
    upper_heter_ff = config_heter_ff * 1.2

    def create_samples_from_config():
        J_array = (upper_J - lower_J) * torch.rand(4, device=device) + lower_J
        P_array = (upper_P - lower_P) * torch.rand(4, device=device) + lower_P
        w_array = (upper_w - lower_w) * torch.rand(4, device=device) + lower_w
        heter_ff = (upper_heter_ff - lower_heter_ff) * torch.rand(1, device=device) + lower_heter_ff
        return J_array, P_array, w_array, heter_ff


    def create_valid_samples():
        bessel_val = torch.tensor(1, device=device)
        i = 0
        while bessel_val != 0 or i > 10000:
            J_array, P_array, w_array, heter_ff = create_samples_from_config()
            wg = WeightsGenerator(J_array, P_array, w_array, 1000, device=device, forward_mode=True)
            bessel_val = wg.validate_weight_matrix()
            i += 1
        return J_array, P_array, w_array, heter_ff


    def create_tuning_curves(J_array, P_array, w_array, heter_ff):
        J_array = torch.tensor(J_array, device=device)
        P_array = torch.tensor(P_array, device=device)
        w_array = torch.tensor(w_array, device=device)
        wg = WeightsGenerator(J_array, P_array, w_array, 1000, device=device, forward_mode=True)
        W = wg.generate_weight_matrix()
        executer.update_heter_ff(heter_ff)
        tuning_curves, _ = executer.run_all_orientation_and_contrast(W)
        y_E, y_I = tuning_curves[:800], tuning_curves[800:]
        return y_E, y_I
    
    results = []
    for _ in range(dataset_trials):
        trial_result = {}
        J_truth, P_truth, w_truth, heter_ff_truth = create_valid_samples()
        trial_result["J_array"] = J_truth.clone().detach().to("cpu")
        trial_result["P_array"] = P_truth.clone().detach().to("cpu")
        trial_result["w_array"] = w_truth.clone().detach().to("cpu")
        trial_result["heter_ff"] = heter_ff_truth.clone().detach().to("cpu")

        y_E, y_I = create_tuning_curves(J_truth, P_truth, w_truth, heter_ff_truth)
        lowest_lost = 10000
        J_lowest = None
        P_lowest = None
        w_lowest = None
        for _ in range(restart_num):
            J_array = torch.rand(4) * 9 - 4.5
            P_array = torch.rand(4) * 9 - 4.5
            w_array = torch.rand(4) * 9 - 4.5
            J_array = torch.tensor(J_array, device= device, requires_grad=True)
            P_array = torch.tensor(P_array, device= device, requires_grad=True)
            w_array = torch.tensor(w_array, device= device, requires_grad=True)
            heter_ff = torch.rand((1), device=device, requires_grad=True)

            loss_diffs = []
            prev_loss = torch.tensor(10000, device=device)
            stopping_criterion_count = 0
            file_name = f"{directory_name}/log_method_val_{time.time()}.log"
            for i in range(200):
                wg = WeightsGenerator(J_array, P_array, w_array, 1000, device=device, forward_mode=True)

                W = wg.generate_weight_matrix()
                executer.update_heter_ff(heter_ff)
                tuning_curves, avg_step = executer.run_all_orientation_and_contrast(W)
                x_E, x_I = tuning_curves[:800], tuning_curves[800:]
                bessel_val = wg.validate_weight_matrix()
                trial_loss, trial_mmd_loss = loss_function.calculate_loss(x_E, y_E, x_I, y_I, avg_step, bessel_val=bessel_val)
                trial_loss.backward()

                # GD
                J_array: torch.Tensor = (J_array - 1 * wg.J_parameters.grad).clone().detach().requires_grad_(True)
                P_array: torch.Tensor = (P_array - 1 * wg.P_parameters.grad).clone().detach().requires_grad_(True)
                w_array: torch.Tensor = (w_array - 1 * wg.w_parameters.grad).clone().detach().requires_grad_(True)
                heter_ff: torch.Tensor  = (heter_ff - 1 * heter_ff.grad).clone().detach().requires_grad_(True)

                loss_diffs.append(prev_loss - trial_mmd_loss.clone().detach())

                if i > 40 and torch.tensor(loss_diffs[-10:], device=device).mean() < 1e-5:
                    if stopping_criterion_count >= 2:
                        break
                    stopping_criterion_count += 1
                else:
                    stopping_criterion_count = 0
                prev_loss = trial_mmd_loss.clone().detach()
                
                # Terminate if nan
                if torch.isnan(J_array).any().item() or torch.isnan(P_array).any().item() or torch.isnan(w_array).any().item() or torch.isnan(heter_ff).any().item():
                    break

                if trial_loss < lowest_lost:
                    lowest_lost = trial_loss.clone().detach().to("cpu")
                    J_lowest = J_array.clone().detach().to("cpu")
                    P_lowest = P_array.clone().detach().to("cpu")
                    w_lowest = w_array.clone().detach().to("cpu")
                    heter_ff_lowest = heter_ff.clone().detach().to("cpu")

        if J_lowest is not None:
            trial_result["J_predicted"] = J_lowest
            trial_result["P_predicted"] = P_lowest
            trial_result["w_predicted"] = w_lowest
            trial_result["heter_ff_predicted"] = heter_ff_lowest
            trial_result["loss"] = lowest_lost

        results.append(trial_result)
    
    with open("method_validation_multi_dataset.pkl", 'wb') as f:
        pickle.dump(results, f)
