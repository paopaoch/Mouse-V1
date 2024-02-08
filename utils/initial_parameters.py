from scipy.special import i0
import math

# TODO: PUT THIS WHOLE THING INSIDE 2 CLASSES, ONE CALLED CONNECTIONS AND THE OTHER CALL CONDITIONS


def calc_theoretical_weights_tot(J, P, w, N_b):
    k = 1 / (4 * (w * math.pi / 180) ** 2)
    bessel = i0(k)
    return J * math.sqrt(N_b) * P * math.exp(-k) * bessel


def first_condition(J_EE, J_EI, J_IE, J_II):
    if not (J_EI < J_EE):
        print("First condition, first inequality")
        return False
    if not (J_EE < J_II):
        print("First condition, second inequality")
        return False
    if not (J_II < J_IE):
        print("First condition, third inequality")
        return False
    return True


def second_condition(w_EE, w_EI, w_IE, w_II):
    if not (w_EI < w_EE):
        print("Second condition, first inequality")
        return False
    if not (w_EE == w_II):  # TODO: Change this to a soft condition
        print("Second condition, second inequality")
        return False
    if not (w_II < w_IE):
        print("Second condition, third inequality")
        return False
    return True


def third_condition(W_tot_EE, W_tot_EI, W_tot_IE, W_tot_II):
    if not ((W_tot_EE / W_tot_IE) < (W_tot_EI / W_tot_II)):
        print("Third condition, first inequality")
        return False
    if not ((W_tot_EI / W_tot_II) < 1):
        print("Third condition, second inequality")
        return False
    return True


def _sigmoid(value, steepness=1, scaling=1):
    return scaling / (1 + math.exp(-steepness * value))


def _inverse_sigmoid(value, steepness=1, scaling=1):
    return - (1 / steepness) * math.log((scaling / value) - 1)


J_to_params = lambda x: _inverse_sigmoid(x, 1/4, 4)
P_to_params = lambda x: _inverse_sigmoid(x, 1/3, 1)
w_to_params = lambda x: _inverse_sigmoid(x, 1/180, 180)


if __name__ == "__main__":

    INIT_PARAMS = True

    if INIT_PARAMS:
        J_EE = 2
        J_EI = 1.5
        J_IE = 3
        J_II = 2.5

        P_EE = 0.2
        P_EI = 0.7
        P_IE = 0.5
        P_II = 0.5

        w_EE = 65
        w_EI = 50
        w_IE = 80
        w_II = 65

    else:
        mean_list = [ 6.7310e-02, -2.1517e+00,  4.5293e+00,  2.0414e+00, -4.1712e+00,
                2.5880e+00, -6.7668e-02, -6.4035e-02, -1.0263e+02, -1.7228e+02,
                -4.0358e+01, -1.0280e+02]

        J_EE = _sigmoid(mean_list[0], 1/4, 4)
        J_EI = _sigmoid(mean_list[1], 1/4, 4)
        J_IE = _sigmoid(mean_list[2], 1/4, 4)
        J_II = _sigmoid(mean_list[3], 1/4, 4)

        P_EE = _sigmoid(mean_list[4], 1/3, 1)
        P_EI = _sigmoid(mean_list[5], 1/3, 1)
        P_IE = _sigmoid(mean_list[6], 1/3, 1)
        P_II = _sigmoid(mean_list[7], 1/3, 1)

        w_EE = _sigmoid(mean_list[8], 1/180, 180)
        w_EI = _sigmoid(mean_list[9], 1/180, 180)
        w_IE = _sigmoid(mean_list[10], 1/180, 180)
        w_II = _sigmoid(mean_list[11], 1/180, 180)

    W_tot_EE = calc_theoretical_weights_tot(J_EE, P_EE, w_EE, 8000)
    W_tot_EI = calc_theoretical_weights_tot(J_EI, P_EI, w_EI, 2000)
    W_tot_IE = calc_theoretical_weights_tot(J_IE, P_IE, w_IE, 8000)
    W_tot_II = calc_theoretical_weights_tot(J_II, P_II, w_II, 2000)

    print(first_condition(J_EE, J_EI, J_IE, J_II))
    print(second_condition(w_EE, w_EI, w_IE, w_II))
    print(third_condition(W_tot_EE, W_tot_EI, W_tot_IE, W_tot_II))
    
    print([J_EE, J_EI, J_IE, J_II, P_EE ,P_EI ,P_IE , P_II, w_EE ,w_EI ,w_IE ,w_II])
    print("mean_list =", [J_to_params(J_EE), J_to_params(J_EI), J_to_params(J_IE), J_to_params(J_II), 
        P_to_params(P_EE), P_to_params(P_EI), P_to_params(P_IE), P_to_params(P_II),  
        w_to_params(w_EE), w_to_params(w_EI), w_to_params(w_IE), w_to_params(w_II)])
    
    print("J_array =", [J_to_params(J_EE), J_to_params(J_EI), J_to_params(J_IE), J_to_params(J_II)])
    print("P_array =", [P_to_params(P_EE), P_to_params(P_EI), P_to_params(P_IE), P_to_params(P_II)])
    print("w_array =", [w_to_params(w_EE), w_to_params(w_EI), w_to_params(w_IE), w_to_params(w_II)])