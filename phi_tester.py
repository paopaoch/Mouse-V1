from rat import NetworkExecuter
import matplotlib.pyplot as plt
import torch


# for i in range(1, 100, 20):
#     executer = NetworkExecuter(10000)
#     mu, sigma = executer._stim_to_inputs(1, 45)
#     executer.hardness = i  # max uses 50
#     executer.mu = mu
#     executer.sigma = sigma

#     response_exact = executer._phi()
#     response_approx = executer._phi_approx()

#     plt.title(f"Exact and approximation curve for hardness={i}")
#     plt.plot(response_exact)
#     plt.plot(response_approx)
#     plt.show()

# -------------------------------------

# for i in range(1, 100, 20):
#     executer = NetworkExecuter(10000)
#     x = torch.linspace(-100, 100, 10000)
#     executer.hardness = i  # max uses 50
#     executer.mu = x
#     executer.sigma = 8

#     response_exact = executer._phi()
#     response_approx = executer._phi_approx()

#     plt.plot(x, response_exact, label="function _phi()")
#     plt.plot(x, response_approx, label="function _phi_approx()")
#     plt.legend()
#     plt.show()


# -------------------------------------


# executer = NetworkExecuter(10000)
# mu, sigma = executer._stim_to_inputs(1, 45)
# executer.hardness = 0.01  # max uses 50
# executer.mu = mu
# executer.sigma = sigma

# response_exact = executer._phi()
# response_approx = executer._phi_approx()

# x = torch.linspace(-100, 100, 10000)
# plt.title(f"Exact and approximation curve for hardness={executer.hardness}")
# plt.plot(x, response_exact, label="function _phi()")
# plt.plot(x, response_approx, label="function _phi_approx()")
# plt.legend()
# plt.show()

# x = torch.linspace(-100, 100, 10000)
# executer.hardness = 0.01  # max uses 50
# executer.mu = x
# executer.sigma = 8

# response_exact = executer._phi()
# response_approx = executer._phi_approx()

# plt.title(f"Exact and approximation curve for hardness={executer.hardness}")
# plt.plot(x, response_exact, label="function _phi()")
# plt.plot(x, response_approx, label="function _phi_approx()")
# plt.legend()
# plt.show()


# -------------------------------------


# executer = NetworkExecuter(10000)
# mu, sigma = executer._stim_to_inputs(1, 45)
# executer.hardness = 0.01  # max uses 50
# executer.mu = mu
# executer.sigma = sigma

# response_exact = executer._phi()
# response_approx = executer._phi_approx_relu_step()

# x = torch.linspace(-100, 100, 10000)
# plt.plot(x, response_exact, label="function _phi()")
# plt.plot(x, response_approx, label="function _phi_approx()")
# plt.legend()
# plt.show()

# x = torch.linspace(-100, 100, 10000)
# executer.hardness = 0.01  # max uses 50
# executer.mu = x
# executer.sigma = 8

# response_exact = executer._phi()
# response_approx = executer._phi_approx_relu_step()

# plt.plot(x, response_exact, label="function _phi()")
# plt.plot(x, response_approx, label="function _phi_approx()")
# plt.legend()
# plt.show()


# -------------------------------------


executer = NetworkExecuter(1000)
mu, sigma = executer._stim_to_inputs(1, 45)
mu = mu.requires_grad_(True)
executer.mu = mu
executer.sigma = sigma

response_approx = executer._phi_approx_relu_step()

output = response_approx.sum()

output.backward()
print(mu.grad)


# -------------------------------------


executer = NetworkExecuter(1000)
mu, sigma = executer._stim_to_inputs(1, 45)
mu = mu.requires_grad_(True)
executer.mu = mu
executer.sigma = sigma

response_approx = executer._phi()

output = response_approx.sum()

output.backward()
print(mu.grad)


# -------------------------------------


executer = NetworkExecuter(1000)
mu, sigma = executer._stim_to_inputs(1, 45)
mu = mu.requires_grad_(True)
executer.mu = mu
executer.sigma = sigma
executer.hardness = 21

response_approx = executer._phi_approx()

output = response_approx.sum()

output.backward()
print(mu.grad)