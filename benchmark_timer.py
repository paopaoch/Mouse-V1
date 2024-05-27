import torch
from rat import get_data, WeightsGenerator, NetworkExecuter, MouseLossFunction
import time
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

if torch.cuda.is_available():
    device = "cuda:1"
    print("Model will be created on GPU")
else:
    device = "cpu"  # MPS is too slow
    print("GPU not available. Model will be created on CPU.")

X, Y = get_data(device=device)
n = 1000
n_E = 800
n_ff = 1000

J_array = [-13.862943611198906, -24.423470353692043, -6.1903920840622355, -24.423470353692043, -24.423470353692043, -24.423470353692043]
P_array = [-2.5418935811616112, 4.1588830833596715, -2.5418935811616112, 4.1588830833596715, -2.5418935811616112, -2.5418935811616112]
w_array = [-237.28336161747745, -255.84942256760897, -214.12513203729057, -225.49733432916625, -255.84942256760897, -255.84942256760897] 


WG = WeightsGenerator(J_array, P_array, w_array, n, n_ff, device=device)
NE = NetworkExecuter(n, n_ff, device=device)
loss_function = MouseLossFunction(device=device)

start = time.time()
print("Validate Weight Matrix:", WG.validate_weight_matrix())
recurrent_weights = WG.generate_weight_matrix()
feed_forward_weights = WG.generate_feed_forward_weight_matrix()
print("Generate weights took:", time.time() - start)

start = time.time()
response, avg_step = NE.run_all_orientation_and_contrast(weights=recurrent_weights, weights_FF=feed_forward_weights)
print("Get all tuning curves took:", time.time() - start)

start = time.time()
loss = loss_function.calculate_loss(response[:n_E], X, response[n_E:], Y, avg_step)
print("Calculating loss took:", time.time() - start)
