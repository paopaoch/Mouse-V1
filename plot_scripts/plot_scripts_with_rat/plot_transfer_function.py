from rat import NetworkExecuter
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.rcParams.update({'font.size': 16})
rcParams.update({'figure.autolayout': True})

executer = NetworkExecuter(100)
sigmas = [1, 2, 3, 4, 5, 6]

for sigma in sigmas:
    executer.mu = torch.linspace(-10, 40, 100)
    executer.sigma = sigma
    y = executer._phi()

    plt.plot(executer.mu, y, label=f"$\sigma$={sigma}")

plt.legend()
plt.xlabel("Input mean")
plt.ylabel("Output response")
plt.show()