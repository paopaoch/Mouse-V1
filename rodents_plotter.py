import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
from rat import get_data, WeightsGeneratorExact, NetworkExecuterParallel
from scipy.stats import circvar
import os
import pickle

if __name__ == "__main__":
    SHOW = bool(input("Enter for save plots: "))
    FOLDER_NAME = f"./plots/ignore_plots_{time.time()}"
    if not SHOW:
        os.mkdir(FOLDER_NAME)
else:
    SHOW = True
    FOLDER_NAME = None


def plot_weights(W: torch.Tensor):
    W = W.clone().detach().cpu()
    plt.imshow(W, cmap="seismic", vmin=-np.max(np.abs(np.array(W))), vmax=np.max(np.abs(np.array(W))), interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title(f"Connection weight matrix for {len(W)} by {len(W[0])} neurons")
    plt.xlabel("Neuron index")
    plt.ylabel("Neuron index")
    if SHOW:
        plt.show()
    else:
        plt.savefig(f"{FOLDER_NAME}/weights_{time.time()}.png")
        plt.close()


def print_tuning_curve(tuning_curve, title=""):
    if type(tuning_curve) == torch.Tensor:
        tuning_curve = np.array(tuning_curve.data)

    plt.imshow(tuning_curve, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("orientation index")
    plt.ylabel("contrast index")
    if SHOW:
        plt.show()
    else:
        plt.savefig(f"{FOLDER_NAME}/tuning_curve_image_{time.time()}.png")
        plt.close()

    for c in tuning_curve:
        plt.plot(c)
        plt.title(title)
        plt.xlabel("orientation index")
        plt.ylabel("Responses")

    if SHOW:
        plt.show()
    else:
        plt.savefig(f"{FOLDER_NAME}/tuning_curve_{time.time()}.png")
        plt.close()


def print_feed_forward_input(executer: NetworkExecuterParallel, W, W_FF):
    executer.update_weight_matrix(W, W_FF)
    # mean, sigma = executer._stim_to_inputs_with_ff(1, 45)
    mean, sigma = executer._stim_to_inputs_with_ff()
    mean = mean[0].cpu()
    sigma = sigma[0].cpu()
    plt.plot(mean)
    plt.title("Mean feed forward activity")
    plt.xlabel("Neuron Index")
    plt.ylabel("Response / Hz")
    if SHOW:
        plt.show()
    else:
        plt.savefig(f"{FOLDER_NAME}/feed_forward_mean_activity_{time.time()}.png")
        plt.close()

    plt.plot(sigma)
    plt.title("Standard deviation of the feed forward activity")
    plt.xlabel("Neuron Index")
    plt.ylabel("Response / Hz")
    if SHOW:
        plt.show()
    else:
        plt.savefig(f"{FOLDER_NAME}/feed_forward_std_activity_{time.time()}.png")
        plt.close()


def print_activity(responses, title="", contrast_index=7):
    one_res = []
    for tuning_curve in responses:
        one_res.append(tuning_curve[contrast_index][4])

    plt.plot(one_res)
    plt.title(title)
    plt.xlabel("Neuron Index")
    plt.ylabel("Response / Hz")
    if SHOW:
        plt.show()
    else:
        plt.savefig(f"{FOLDER_NAME}/activity_{time.time()}.png")
        plt.close()


def neuro_SVD(tuning_curve):
    U, S, Vt = np.linalg.svd(tuning_curve)
    k = 1 # number of singular values to keep
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    reduced_neuron = np.dot(U_k, np.dot(S_k, Vt_k))

    residue = tuning_curve - reduced_neuron
    return reduced_neuron, residue, S


def get_all_percentage_explained(responses):
    percentages = []
    for tuning_curve in responses:
        _, residue, S = neuro_SVD(tuning_curve)
        percentage_explained = (np.sum(residue)**2) / (np.sum(tuning_curve)**2)
        percentages.append(percentage_explained)
    return percentages


def get_all_fraction_of_variance(responses):
    frac_of_vars = []
    for tuning_curve in responses:
        _, _, S = neuro_SVD(tuning_curve)
        # frac_of_var = (np.linalg.norm(S[1:]) ** 2) / (np.linalg.norm(S) ** 2)
        frac_of_var = (np.linalg.norm(S[0]) ** 2) / (np.linalg.norm(S) ** 2)
        frac_of_vars.append(frac_of_var)
    return frac_of_vars


def plot_percentage_explained(tuning_curves, title=""):
    percentages = get_all_percentage_explained(tuning_curves)
    plt.hist(percentages, 10)
    plt.title(title)
    if SHOW:
        plt.show()
    else:
        plt.savefig(f"{FOLDER_NAME}/percentage_{time.time()}.png")
        plt.close()


def plot_frac_of_var(tuning_curves, title="", bin_size=0.0025, xlim=0.85):
    frac_of_vars = get_all_fraction_of_variance(tuning_curves)
    bins = np.arange(0, 1 + bin_size, bin_size)
    plt.hist(frac_of_vars, bins, bottom=0, width=bin_size, color="cadetblue")
    plt.xticks(np.arange(0, 1 + bin_size, bin_size * 20))
    plt.xlim(xlim, 1)
    plt.title(title)
    plt.xlabel("Fraction of data explained by first SV")
    plt.ylabel("Unit count")
    if SHOW:
        plt.show()
    else:
        plt.savefig(f"{FOLDER_NAME}/frac_of_var_{time.time()}.png")
        plt.close()


def get_circ_var(tuning_curve, contrast_index=7):
    tc_1D = tuning_curve[contrast_index]
    return circvar(tc_1D)


def get_max_firing_rate(tuning_curve, contrast_index=None):
    return np.max(tuning_curve)


def get_mean_firing_rate(tuning_curve, contrast_index=None):
    return np.mean(tuning_curve)


def plot_hist(func, responses, contrast_index=7, title="", bin_size=None, bin_num=20):
    circ_vars = []

    for response in responses:
        circ_vars.append(func(response, contrast_index))

    if bin_size is not None:
        bins = np.arange(min(circ_vars), max(circ_vars) + bin_size, bin_size)
        plt.hist(circ_vars, bins)
    else:
        plt.hist(circ_vars, bin_num)
        
    plt.title(title)
    if SHOW:
        plt.show()
    else:
        plt.savefig(f"{FOLDER_NAME}/hist_{time.time()}.png")
        plt.close()


def get_max_index(tuning_curve):
    max_index = np.argmax(tuning_curve[7])
    return max_index


def normalise_array(arr, scale=1):
    max_value = max(arr)
    if max_value == 0:
        return arr
    else:
        normalised_arr = arr / max_value
        return normalised_arr * scale


def print_normalise_orientation_curve(tuning_curve):
    max_rate = max(tuning_curve[7])
    for i, orientation_tuning in enumerate(tuning_curve):
        orientation_tuning_normalised = normalise_array(orientation_tuning, max_rate)
        plt.plot(orientation_tuning_normalised, label=f"contrast index {i}")
    
    plt.title("Normalised orientation tuning curves at various contrast of the same neuron")
    plt.xlabel("orientation index")
    plt.ylabel("rate/Hz")
    plt.legend()
    if SHOW:
        plt.show()
    else:
        plt.savefig(f"{FOLDER_NAME}/normalise_orientation_curve_{time.time()}.png")
        plt.close()


def print_normalise_orientation_curve_multi_neuron(tuning_curves, contrast_index=7, title=""):
    max_rate = np.max(tuning_curves)
    for tuning_curve in tuning_curves:
        orientation_tuning_normalised = normalise_array(tuning_curve[contrast_index], max_rate)
        plt.plot(orientation_tuning_normalised)
    
    plt.title(title)
    plt.xlabel("orientation index")
    plt.ylabel("rate/Hz")
    if SHOW:
        plt.show()
    else:
        plt.savefig(f"{FOLDER_NAME}/normalise_orientation_curve_multi_neuron{time.time()}.png")
        plt.close()


def print_contrast_curve(tuning_curves: list, title=""):
    for tuning_curve in tuning_curves:
        tuning_curve.transpose(1, 0)
        plt.plot(tuning_curve.transpose(1, 0)[6])
    
    plt.title(title)
    plt.xlabel("orientation index")
    plt.ylabel("rate/Hz")
    if SHOW:
        plt.show()
    else:
        plt.savefig(f"{FOLDER_NAME}/contrast_tuning_curve{time.time()}.png")
        plt.close()


def centralise_curve(tuning_curve):
    max_index = get_max_index(tuning_curve)  # instead of max index, taking the mean might be better?
    shift_index = 6 - max_index  # 6 is used here as there are 13 orientations
    new_tuning_curve = np.roll(tuning_curve, int(shift_index), axis=1)
    return new_tuning_curve


def centralise_all_curves(responses):
    tuning_curves = []
    for tuning_curve in responses:
        tuning_curves.append(centralise_curve(tuning_curve))
    return np.stack(tuning_curves)


if __name__ == "__main__":

    not_data = bool(input("Enter for plotting data: "))


    if not_data:
        responses_path = input("Path to response file: ")
        if responses_path == "":
            neuron_num = 1000
            ratio = 0.8
            E_index = int(ratio * neuron_num)
            feed_forward_num = 1000 
            scaling_g = 0.15  # without ff use 0.15

            # Get the network response

            # J_array = [-2.059459853260332, -3.0504048076264896, -1.5877549090278045, -2.813481385641024]  # n = 1000
            # P_array = [-2.0907410969337694, -0.20067069546215124, -2.0907410969337694, -0.20067069546215124]
            # w_array = [-1.5314763709643886, -1.5314763709643886, -1.5314763709643886, -1.5314763709643886] 

            # J_array = [-1.6582280766035324, -2.68273239311792, -1.1526795099383855, -2.4423470353692043]  # n = 2000
            # P_array = [-2.0907410969337694, -0.20067069546215124, -2.0907410969337694, -0.20067069546215124]
            # w_array = [-1.5314763709643886, -1.5314763709643886, -1.5314763709643886, -1.5314763709643886] 

            # J_array = [-1.411484609948449, -2.4642873625852646, -0.8765172970974294, -2.2196470413920517]  # n = 3000
            # P_array = [-2.0907410969337694, -0.20067069546215124, -2.0907410969337694, -0.20067069546215124]
            # w_array = [-1.5314763709643886, -1.5314763709643886, -1.5314763709643886, -1.5314763709643886] 

            # J_array = [-0.5849256105807946, -1.7897775391126678, 0.1470644867601804, -1.5238179142000694]  # n = 10000
            # P_array = [-2.0907410969337694, -0.20067069546215124, -2.0907410969337694, -0.20067069546215124]
            # w_array = [-1.5314763709643886, -1.5314763709643886, -1.5314763709643886, -1.5314763709643886] 

            # J_array = [-0.5849256105807946, -1.7897775391126678, 0.1470644867601804, -1.5238179142000694, -2.9444389791664403, -2.9444389791664403]
            # P_array = [-2.0907410969337694, -0.20067069546215124, -2.0907410969337694, -0.20067069546215124, -2.0907410969337694, -2.0907410969337694]
            # w_array = [-1.5314763709643886, -1.5314763709643886, -1.5314763709643886, -1.5314763709643886, -1.6094379124341003, -1.6094379124341003] 
            
            J_array = [0.587113, -1.006514, -1.494353, -0.07066000000000026]
            P_array = [-3.455629, -0.830209, 0.48249700000000006, 1.6340260000000002]
            w_array = [0.6527120000000001, 1.989928, -0.7805890000000001, 0.11133499999999995]

            generator = WeightsGeneratorExact(J_array, P_array, w_array, neuron_num, feed_forward_num)
            W = generator.generate_weight_matrix()
            plot_weights(W)
            if len(J_array) == 6:
                W_FF = generator.generate_feed_forward_weight_matrix()
                plot_weights(W_FF)
            else:
                W_FF = None

            executer = NetworkExecuterParallel(neuron_num, feed_forward_num, scaling_g=scaling_g, device="cpu", sig_ext=5, w_ff=30)
            if len(J_array) == 6:
                print_feed_forward_input(executer, W, W_FF)

            responses, _ = executer.run_all_orientation_and_contrast(W, W_FF)
            
            if not SHOW:
                with open(f"{FOLDER_NAME}/responses.pkl", "wb") as f:
                    pickle.dump(responses, f)
                with open(f"{FOLDER_NAME}/plot_log.log", 'w') as f:
                    f.write(f"PLOT LOG FILE FOR {datetime.now()}\n\n")
                    f.write(f"J_array = {J_array}\n")
                    f.write(f"P_array = {P_array}\n")
                    f.write(f"w_array = {w_array}\n")
            
        else:
            with open(responses_path, 'rb') as f:
                responses = pickle.load(f)
            
            if type(responses) != torch.Tensor:
                responses = torch.tensor(responses)
        
        responses = responses.cpu()

        data_E = centralise_all_curves(np.array(responses[0:E_index].data))
        data_I = centralise_all_curves(np.array(responses[E_index:].data))
        data = np.concatenate((data_E, data_I), axis=0)

        # print_normalise_orientation_curve(data[100])
        print_normalise_orientation_curve_multi_neuron([data[100], data[150], data[200], data[250]]
                                                       , 7, f"Normalised excitatory orientation tuning curves \n at constant contrast index 7 for multiple neurons")
        print_normalise_orientation_curve_multi_neuron([data[100], data[150], data[200], data[250]]
                                                       , 5, f"Normalised excitatory orientation tuning curves \n at constant contrast index 5 for multiple neurons")
        print_normalise_orientation_curve_multi_neuron([data[100], data[150], data[200], data[250]]
                                                       , 3, f"Normalised excitatory orientation tuning curves \n at constant contrast index 3 for multiple neurons")
        print_contrast_curve([data[100], data[150], data[200], data[250]], "Normalised excitatory contrast tuning curves at preferred orientation")

        print_normalise_orientation_curve_multi_neuron([data[800], data[850], data[900], data[950]]
                                                       , 7, f"Normalised inhibitory orientation tuning curves \n at constant contrast index 7 for multiple neurons")
        print_normalise_orientation_curve_multi_neuron([data[800], data[850], data[900], data[950]]
                                                       , 5, f"Normalised inhibitory orientation tuning curves \n at constant contrast index 5 for multiple neurons")
        print_normalise_orientation_curve_multi_neuron([data[800], data[850], data[900], data[950]]
                                                       , 3, f"Normalised inhibitory orientation tuning curves \n at constant contrast index 3 for multiple neurons")
        print_contrast_curve([data[800], data[850], data[900], data[950]], "Normalised inhibitory contrast tuning curves at preferred orientation")

        print_tuning_curve(data[100], title="Example Excitatory Neuron Tuning Curve From Model")
        print_tuning_curve(data[-100], title="Example Inhibitory Neuron Tuning Curve From Model")
        
        print_activity(responses[:E_index], title="Example Response Plot for the Model \n Excitatory (High contrast)", contrast_index=7)
        print_activity(responses[E_index:], title="Example Response Plot for the Model \n Inhibitory (High contrast)", contrast_index=7)

        print_activity(responses[:E_index], title="Example Response Plot for the Model \n Excitatory (Mid contrast)", contrast_index=5)
        print_activity(responses[E_index:], title="Example Response Plot for the Model \n Inhibitory (Mid contrast)", contrast_index=5)

        print_activity(responses[:E_index], title="Example Response Plot for the Model \n Excitatory (Low contrast)", contrast_index=3)
        print_activity(responses[E_index:], title="Example Response Plot for the Model \n Inhibitory (Low contrast)", contrast_index=3)

        print_tuning_curve(neuro_SVD(data[100])[0], title="Example SVD of Excitatory Neuron Tuning Curve From Model")
        print_tuning_curve(neuro_SVD(data[-100])[0], title="Example SVD of Inhibitory Neuron Tuning Curve From Model")

        plot_percentage_explained(data, title="Histogram of the percentage that the residue is left after SVD")

        plot_frac_of_var(data_E, title="Fraction of explained variance (degree of contrast invariance) - Excitatory")
        plot_frac_of_var(data_I, title="Fraction of explained variance (degree of contrast invariance) - Inhibitory")

        plot_frac_of_var(data_E, title="Full fraction of explained variance (degree of contrast invariance) - Excitatory", xlim=0)
        plot_frac_of_var(data_I, title="Full fraction of explained variance (degree of contrast invariance) - Inhibitory", xlim=0)

        plot_hist(get_circ_var, data_E, title="Histogram of the circular variance of the model E tuning curves")
        plot_hist(get_max_firing_rate, data_E, title="Histogram of the max firing rate of the model E tuning curves")
        plot_hist(get_mean_firing_rate, data_E, title="Histogram of the mean firing rate of the model E tuning curves")

        plot_hist(get_circ_var, data_I, title="Histogram of the circular variance of the model I tuning curves")
        plot_hist(get_max_firing_rate, data_I, title="Histogram of the max firing rate of the model I tuning curves")
        plot_hist(get_mean_firing_rate, data_I, title="Histogram of the mean firing rate of the model I tuning curves")

    # else:
        # # Get the data
        # data_E, data_I = get_data()
        # responses = np.concatenate((np.array(data_E.data), np.array(data_I.data)), axis=0)
        # data_E = centralise_all_curves(np.array(data_E.data))
        # data_I = centralise_all_curves(np.array(data_I.data))
        # data = np.concatenate((data_E, data_I), axis=0)

        # print_tuning_curve(data[5], title="Example Excitatory Neuron Tuning Curve From Data")
        # print_tuning_curve(data[-5], title="Example Inhibitory Neuron Tuning Curve From Data")
        
        # print_activity(responses, title="Response Plot for the data")

        # print_tuning_curve(neuro_SVD(data[5])[0], title="Example SVD of Excitatory Neuron Tuning Curve From Data")
        # print_tuning_curve(neuro_SVD(data[-5])[0], title="Example SVD of Inhibitory Neuron Tuning Curve From Data")

        # plot_percentage_explained(data, title="Histogram of the percentage that the residue is left after SVD")

        # plot_frac_of_var(data_E, title="Fraction of explained variance (degree of contrast invariance) - Excitatory")
        # plot_frac_of_var(data_I, title="Fraction of explained variance (degree of contrast invariance) - Inhibitory")

        # plot_hist(get_circ_var, data_E, title="Histogram of the circular variance of the data E tuning curves")
        # plot_hist(get_max_firing_rate, data_E, title="Histogram of the max firing rate of the data E tuning curves")
        # plot_hist(get_mean_firing_rate, data_E, title="Histogram of the mean firing rate of the data E tuning curves")

        # plot_hist(get_circ_var, data_I, title="Histogram of the circular variance of the data I tuning curves")
        # plot_hist(get_max_firing_rate, data_I, title="Histogram of the max firing rate of the data I tuning curves")
        # plot_hist(get_mean_firing_rate, data_I, title="Histogram of the mean firing rate of the data I tuning curves")
