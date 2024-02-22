import torch
from rat import get_data, WeightsGenerator, NetworkExecuter, MouseLossFunction
import time
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = "cuda:1"
    print("Model will be created on GPU")
else:
    device = "cpu"
    print("GPU not available. Model will be created on CPU.")

X, Y = get_data(device=device)
n = 10000
n_E = 8000

J_array = [-13.862943611198906, -24.423470353692043, -6.1903920840622355, -24.423470353692043, -24.423470353692043, -24.423470353692043]
P_array = [-2.5418935811616112, 4.1588830833596715, -2.5418935811616112, 4.1588830833596715, -2.5418935811616112, -2.5418935811616112]
w_array = [-237.28336161747745, -255.84942256760897, -214.12513203729057, -225.49733432916625, -255.84942256760897, -255.84942256760897] 


# J_array = [-3.9719, -15.8767, 9.1343, -14.8888, 0, 0]
# P_array = [-11.3945, -1.0984, -9.5121, -2.8681, 0, 0]
# w_array = [-256.1385, -304.5658, -213.8999, -256.1974, 0, 0]

WG = WeightsGenerator(J_array, P_array, w_array, n, device=device)
NE = NetworkExecuter(n, device=device)
loss_function = MouseLossFunction(device=device)

start = time.time()
recurrent_weights, _ = WG.generate_weight_matrix()
feed_forward_weights = WG.generate_external_weight_matrix()
print("Generate weights took:", time.time() - start)

start = time.time()
response, avg_step = NE.run_all_orientation_and_contrast(weights=recurrent_weights, weights_FF=feed_forward_weights)
print("Get all tuning curves took:", time.time() - start)

start = time.time()
loss = loss_function.calculate_loss(response[:n_E], X, response[n_E:], Y, avg_step)
print("Calculating loss took:", time.time() - start)


# NE.update_weight_matrix(recurrent_weights)
# mean, std = NE._stim_to_inputs(contrast=1, grating_orientation=45)
# plt.plot(mean)
# plt.show()

# NE.update_weight_matrix(recurrent_weights, feed_forward_weights)
# mean, std = NE._stim_to_inputs_with_ff(contrast=1, grating_orientation=45)
# plt.plot(mean)
# plt.show()

# plt.plot(std)
# plt.show()