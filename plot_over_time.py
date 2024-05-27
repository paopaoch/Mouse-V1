from rat import WeightsGeneratorExact, NetworkExecuterWithSimplifiedFF
import torch
import matplotlib.pyplot as plt

n = 10000
plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'figure.autolayout': True})

J_array = [-0.9308613398652443, -2.0604571635972393, -0.30535063458645906, -1.802886963254238]  # config 13
P_array = [-1.493925025312256, 1.09861228866811, -1.493925025312256, 1.09861228866811]
w_array = [-1.5314763709643886, -1.5314763709643886, -1.5314763709643886, -1.5314763709643886] 
heter_ff = torch.tensor([-1.3862943611198906])

wg = WeightsGeneratorExact(J_array, P_array, w_array, n)
executer = NetworkExecuterWithSimplifiedFF(n)

W = wg.generate_weight_matrix()
executer.update_heter_ff(heter_ff)

executer.run_all_orientation_and_contrast(W)

executer.plot_overtime = True

xvec = torch.zeros_like(executer.input_mean)

output, _ = executer._euler2fixedpt(xvec)

print(len(output))
print(output[0].shape)

val1 = []
val2 = []
val3 = []

val4 = []
val5 = []
val6 = []
for item in output:
    val1.append(item[3900][90])
    val2.append(item[4000][90])
    val3.append(item[4100][90])

    val4.append(item[8900][90])
    val5.append(item[9000][90])
    val6.append(item[9100][90])

plt.plot(val1)
plt.plot(val2)
plt.plot(val3)
plt.title("Excitatory")
plt.xlabel("Iterations")
plt.ylabel("rate/Hz")
plt.show()

plt.plot(val4)
plt.plot(val5)
plt.plot(val6)
plt.title("Inihibitory")
plt.xlabel("Iterations")
plt.ylabel("rate/Hz")
plt.show()