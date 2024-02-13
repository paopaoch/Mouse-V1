import pandas as pd
import os

if __name__ == "__main__":
    # file_name = input("path to log file: ")
    file_name = r"playground/log_nes7.log"
    current_dir = os.getcwd()

    losses = {
        "mean" : [],
        "mmd" : [],
        "avg" : [],
        "min" : [],
        "max" : [],
    }

    with open(file_name, "r") as f:
        for line in f:
            words = line.split(" ")
            if words[0] == "Mean":
                losses["mean"].append(float(words[-1]))
            if words[0] == "MMD":
                losses["mmd"].append(float(words[-1]))
            if words[0] == "Avg":
                losses["avg"].append(float(words[-1]))
            if words[0] == "Min":
                losses["min"].append(float(words[-1]))
            if words[0] == "Max":
                losses["max"].append(float(words[-1]))

    df = pd.DataFrame(data=losses)
    df.to_csv(f"{current_dir}/data/losses.csv", index=False)