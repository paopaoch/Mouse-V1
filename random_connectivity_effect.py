from rat import WeightsGenerator, NetworkExecuter, MouseLossFunction
import torch

if torch.cuda.is_available():
    device = "cuda:1"
    print("Model will be created on GPU")
else:
    device = "cpu"
    print("GPU not available. Model will be created on CPU.")

J_array1 = torch.tensor([-3.9440e+00, -1.6740e+01,  8.4091e+00, -1.5425e+01], device=device)
P_array1 = torch.tensor([-9.7804e+00, -1.4716e+00, -9.2311e+00, -1.7742e-01], device=device)
w_array1 = torch.tensor([-2.5519e+02, -3.0496e+02, -2.1453e+02, -2.5578e+02], device=device)

J_array2 = torch.tensor([-4.0547,  -17.3460, 8.4730, -15.8563], device=device)
P_array2 = torch.tensor([-10.9907, -1.2164, -8.8333, -1.2164], device=device)
w_array2 = torch.tensor([-255.8494, -304.5017, -214.1251, -255.8494], device=device)

wg1 = WeightsGenerator(J_array1, P_array1, w_array1, 10000, device=device)
wg2 = WeightsGenerator(J_array2, P_array2, w_array2, 10000, device=device)

executer = NetworkExecuter(10000, device=device)

loss_function = MouseLossFunction(device=device)

with open("loss_comparer.log", "w") as f:
    for i in range(100):
        print(i)
        weights1, _ = wg1.generate_weight_matrix()
        response1, _ = executer.run_all_orientation_and_contrast(weights=weights1)

        weights2, _ = wg2.generate_weight_matrix()
        response2, _ = executer.run_all_orientation_and_contrast(weights=weights2)

        _ , loss = loss_function.calculate_loss(response1[:8000], response2[:8000], response1[8000:], response2[8000:], avg_step=torch.tensor(0))

        f.write(f"{loss}")
        f.flush()
