from rat import WeightsGenerator


J_array = [-22.907410969337693, -32.550488507104102, -17.85627263740382, -30.060150147989074]
P_array = [-3.2163953243244932, 10.833316937499324, -4.2163953243244932, 10.833316937499324]
w_array = [-135.44395575681614, -132.44395575681614, -131.44395575681614, -132.44395575681614]

generator = WeightsGenerator(J_array, P_array, w_array, 1000)

print(generator.calc_theoretical_weights_tot(0, generator.neuron_num_e))
print(generator.calc_theoretical_weights_tot_torch(0, generator.neuron_num_e))

