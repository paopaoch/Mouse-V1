from scipy.special import i0
import math
from rat import WeightsGenerator
from rodents_plotter import plot_weights
import numpy as np

J_steep = 1/10
J_scale = 100

P_steep = 1/3
P_scale = 1

w_steep = 1/180
w_scale = 180

class Connection:
    def __init__(self, J: float, P: float, w: float, N: int):
        self.J = J
        self.P = P
        self.w = w
        self.N = N
        self.W_tot = self.calc_theoretical_weights_tot()

    def calc_theoretical_weights_tot(self):
        k = 1 / (4 * (self.w * math.pi / 180) ** 2)
        bessel = i0(k)
        return self.J * math.sqrt(self.N) * self.P * math.exp(-k) * bessel
    

class ConstraintChecker:
    def __init__(self, EE: Connection, EI: Connection, IE: Connection, II: Connection):
        self.EE = EE
        self.EI = EI
        self.IE = IE
        self.II = II
        self.connections = [self.EE, self.EI, self.IE, self.II]
        self.connection_names = ["EE", "EI", "IE", "II"]
        
        self.wg = WeightsGenerator([J_to_params(EE.J), J_to_params(EI.J), J_to_params(IE.J), J_to_params(II.J)],
                              [P_to_params(EE.P), P_to_params(EI.P), P_to_params(IE.P), P_to_params(II.P)],
                              [w_to_params(EE.w), w_to_params(EI.w), w_to_params(IE.w), w_to_params(II.w)],
                              EE.N + II.N)

        print("Out of bounds found:\n")
        self.kraynyukova_J_condition()
        self.kraynyukova_w_condition()
        self.W_tot_condition()
        self.efficacy_condition()
        self.connection_count_condition()

    def kraynyukova_J_condition(self):
        if not (EI.J / EI.N < EE.J / EI.N):
            print("kraynyukova_J_condition, J_EI/N_I < J_EE/N_E")
            print(EI.J / EI.N, EE.J / EI.N, '\n')
        if not (EE.J / EE.N < II.J / II.N):
            print("kraynyukova_J_condition, J_EE/N_E < J_II/N_I")
            print(EE.J / EE.N, II.J / II.N, '\n')
        if not (II.J / II.N < IE.J / IE.N):
            print("kraynyukova_J_condition, J_II/N_I < J_IE/N_E")
            print(II.J / II.N, IE.J / IE.N, '\n')
    

    def kraynyukova_w_condition(self):
        if not (EI.w < EE.w):
            print("kraynyukova_w_condition, w_EI < w_EE")
            print(EI.w, EE.w, '\n')
        if abs(EE.w - II.w) > 10:  # 10 is a guess as in the paper they use the approx sign
            print("kraynyukova_w_condition, w_EE ~= w_II")
            print(EE.w, II.w, '\n')
        if not (II.w < IE.w):
            print("kraynyukova_w_condition, w_II < w_IE")
            print(II.w, IE.w, '\n')

    
    def W_tot_condition(self):
        if not ((EE.W_tot / IE.W_tot) < (EI.W_tot / II.W_tot)):
            print("W_tot_condition, (W_tot_EE / W_tot_IE) < (W_tot_EI / W_tot_II)")
            print((EE.W_tot / IE.W_tot), (EI.W_tot / II.W_tot), '\n')
        if not ((EI.W_tot / II.W_tot) < 1):
            print("W_tot_condition, (W_tot_EI / W_tot_II) < 1")
            print((EI.W_tot / II.W_tot), 1, '\n')


    def efficacy_condition(self):
        for i, connection in enumerate(self.connections):
            if (connection.J / math.sqrt(connection.N)) < 0.25:
                print(f"efficacy_condition, {self.connection_names[i]} is too low")
                print(connection.J / math.sqrt(connection.N), '\n')

            if (connection.J / math.sqrt(connection.N)) > 2:
                print(f"efficacy_condition, {self.connection_names[i]} is too high")
                print(connection.J / math.sqrt(connection.N), '\n')


    def connection_count_condition(self):
        error = False
        for trial in range(10):  # as weight matrix is random we will test it 10 times
            E_total = 0
            I_total = 0
            weights, _ = self.wg.generate_weight_matrix()
            for i in range(self.EE.N + self.II.N):
                row = np.array(weights[i].data)
                E_total += np.sum(self._normalise(row[:self.EE.N]))
                I_total += np.sum(self._normalise(np.abs(row[self.EE.N:])))

            if E_total / (self.EE.N + self.II.N) > 1200:
                print("Too many E connections")
                print(E_total / (self.EE.N + self.II.N), '\n')
                error = True
            elif E_total / (self.EE.N + self.II.N) < 350:
                print("Too little E connections")
                print(E_total / (self.EE.N + self.II.N), '\n')
                error = True

            if I_total / (self.EE.N + self.II.N) > 1200:
                print("Too many I connections")
                print(I_total / (self.EE.N + self.II.N), '\n')
                error = True
            elif I_total / (self.EE.N + self.II.N) < 350:
                print("Too little I connections")
                print(I_total / (self.EE.N + self.II.N), '\n')
                error = True

            if error and trial > 1:
                return
            


    @staticmethod
    def _normalise(array):
        return (array - np.min(array)) / (np.max(array) - np.min(array))


def _sigmoid(value, steepness=1, scaling=1):
    return scaling / (1 + math.exp(-steepness * value))


def _inverse_sigmoid(value, steepness=1, scaling=1):
    return - (1 / steepness) * math.log((scaling / value) - 1)


J_to_params = lambda x: _inverse_sigmoid(x, J_steep, J_scale)
P_to_params = lambda x: _inverse_sigmoid(x, P_steep, P_scale)
w_to_params = lambda x: _inverse_sigmoid(x, w_steep, w_scale)


params_to_J = lambda x: _sigmoid(x, J_steep, J_scale)
params_to_P = lambda x: _sigmoid(x, P_steep, P_scale)
params_to_w = lambda x: _sigmoid(x, w_steep, w_scale)


def get_random_valid_params():
    return


if __name__ == "__main__":

    INITIAL = bool(input("Initial Params? (default False): "))
    print("\nInitial: ", INITIAL, '\n')
    N_E = 8000
    N_I = 2000

    if INITIAL:
        EE = Connection(30, 0.4, 100, N_E)
        EI = Connection(16, 0.5, 80, N_I)
        IE = Connection(80, 0.2, 110, N_E)
        II = Connection(10, 0.8, 105, N_I)
    else:
        mean_list = [  -5.3943,  -17.2444,    8.9355,  -16.5509,  -10.2303,   -0.6580,
          -8.8399,   -1.5132, -256.3075, -305.1801, -213.9073, -257.2812]
        
        EE = Connection(params_to_J(mean_list[0]), params_to_P(mean_list[4]), params_to_w(mean_list[8]), N_E)
        EI = Connection(params_to_J(mean_list[1]), params_to_P(mean_list[5]), params_to_w(mean_list[9]), N_I)
        IE = Connection(params_to_J(mean_list[2]), params_to_P(mean_list[6]), params_to_w(mean_list[10]), N_E)
        II = Connection(params_to_J(mean_list[3]), params_to_P(mean_list[7]), params_to_w(mean_list[11]), N_I)

    ConstraintChecker(EE, EI, IE, II)

    print("Values in code:\n")

    print([EE.J, EI.J, IE.J, II.J, EE.P ,EI.P ,IE.P , II.P, EE.w, EI.w, IE.w, II.w], '\n')
    print("mean_list =", [J_to_params(EE.J), J_to_params(EI.J), J_to_params(IE.J), J_to_params(II.J), 
            P_to_params(EE.P), P_to_params(EI.P), P_to_params(IE.P), P_to_params(II.P),  
            w_to_params(EE.w), w_to_params(EI.w), w_to_params(IE.w), w_to_params(II.w)], '\n')
    
    J = [J_to_params(EE.J), J_to_params(EI.J), J_to_params(IE.J), J_to_params(II.J)]
    P = [P_to_params(EE.P), P_to_params(EI.P), P_to_params(IE.P), P_to_params(II.P)]
    w = [w_to_params(EE.w), w_to_params(EI.w), w_to_params(IE.w), w_to_params(II.w)]

    print("J_array =", J)
    print("P_array =", P)
    print("w_array =", w, '\n')

    wg = WeightsGenerator(J, P, w, N_E + N_I)
    plot_weights(wg.generate_weight_matrix()[0])