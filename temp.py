import pickle
import torch


with open("method_validation_multi_dataset.pickle", "rb") as f:
    data = pickle.load(f)

results = []
for datum in data:
    temp_datum = {}
    temp_datum["J_predicted"] = datum["J_predicted"].clone().detach().to("cpu")
    temp_datum["P_predicted"] = datum["P_predicted"].clone().detach().to("cpu")
    temp_datum["w_predicted"] = datum["w_predicted"].clone().detach().to("cpu")

    temp_datum["J_array"] = datum["J_array"].clone().detach().to("cpu")
    temp_datum["P_array"] = datum["P_array"].clone().detach().to("cpu")
    temp_datum["w_array"] = datum["w_array"].clone().detach().to("cpu")

    temp_datum["loss"] = datum["loss"].clone().detach().to("cpu")
    results.append(temp_datum)

with open("method_validation_multi_dataset_cpu.pickle", 'wb') as f:
    pickle.dump(results, f)
