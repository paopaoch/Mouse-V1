from rat import WeightsGeneratorExact, OSDependentWeightsGenerator, RandomWeightsGenerator, NetworkExecuterWithSimplifiedFF
from rodents_plotter import plot_weights, print_activity, print_tuning_curve
import matplotlib.pyplot as plt
from matplotlib import rcParams

# J_array = [-0.9308613398652443, -2.0604571635972393, -0.30535063458645906, -1.802886963254238]
# P_array = [-1.493925025312256, 1.09861228866811, -1.493925025312256, 1.09861228866811]
# w_array = [-1.5314763709643886, -1.5314763709643886, -1.5314763709643886, -1.5314763709643886]

# wg = WeightsGeneratorExact(J_array, P_array, w_array, 200)


# W = wg.generate_weight_matrix()

# # plt.figure(figsize=(3, 2), dpi=80)
# plt.rcParams.update({'font.size': 16})
# # plt.tight_layout()
# rcParams.update({'figure.autolayout': True})
# plot_weights(W, title="")


executer = NetworkExecuterWithSimplifiedFF(1000)

executer.update_heter_ff(0.2)

output = executer._stim_to_inputs()

activity = output[0].view(executer.neuron_num, len(executer.contrasts), len(executer.orientations))

mean = activity[400]

plt.rcParams.update({'font.size': 16})
# plt.tight_layout()
rcParams.update({'figure.autolayout': True})

# plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11], orientatio_gap)
# plt.yticks([0,1,2,3,4,5,6,7], contrast_val)
# print_activity(mean[:800])

print_tuning_curve(mean)

# plt.plot(mean)
# plt.show()