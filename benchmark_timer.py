import torch
from rat import get_data, WeightsGenerator, NetworkExecuter, MouseLossFunction
import time

if torch.cuda.is_available():
    device = "cuda:1"
    print("Model will be created on GPU")
else:
    device = "cpu"
    print("GPU not available. Model will be created on CPU.")

X, Y = get_data(device=device)

J_array = [-13.862943611198906, -24.423470353692043, -6.1903920840622355, -24.423470353692043, -13.862943611198906, -13.862943611198906]
P_array = [-2.5418935811616112, 4.1588830833596715, -2.5418935811616112, 4.1588830833596715, -2.5418935811616112, -2.5418935811616112]
w_array = [-237.28336161747745, -255.84942256760897, -214.12513203729057, -225.49733432916625, -237.28336161747745, -237.28336161747745] 

WG = WeightsGenerator(J_array, P_array, w_array, 4000, device=device)
NE = NetworkExecuter(4000)
loss_function = MouseLossFunction()

start = time.time()
recurrent_weights, _ = WG.generate_weight_matrix()
feed_forward_weights = WG.generate_external_weight_matrix()
print("Generate weights took:", time.time() - start)

start = time.time()
response, avg_step = NE.run_all_orientation_and_contrast(weights=recurrent_weights, weights_FF=feed_forward_weights)
print("Get all tuning curves took:", time.time() - start)

start = time.time()
loss = loss_function.calculate_loss(response[:3200], X, response[3200:], Y, avg_step)
print("Calculating loss took:", time.time() - start)