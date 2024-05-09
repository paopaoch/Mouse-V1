from scipy.special import i0
import math
from rat import WeightsGeneratorExact
import numpy as np
from tqdm import tqdm

J_steep = 1
J_scale = 100

P_steep = 1
P_scale = 1

w_steep = 1
w_scale = 180

class Connection:
    def __init__(self, J: float, P: float, w: float, N: int):
        self.J = J
        self.P = P
        self.w = w
        self.N = N
        self.root_N = math.sqrt(N)
        self.W_tot = self.calc_theoretical_weights_tot()


    def calc_theoretical_weights_tot(self):
        k = 1 / (4 * (self.w * math.pi / 180) ** 2)
        bessel = i0(k)
        return self.J * self.root_N * self.P * math.exp(-k) * bessel
    

class ConstraintChecker:
    def __init__(self, EE: Connection, EI: Connection, IE: Connection, II: Connection, EF: Connection=None, IF: Connection=None, print_statement=True, test_trials=1):
        self.EE = EE
        self.EI = EI
        self.IE = IE
        self.II = II
        self.EF = EF
        self.IF = IF
        self.print_statement = print_statement
        self.test_trials = test_trials

        self.connections = [self.EE, self.EI, self.IE, self.II]
        self.connection_names = ["EE", "EI", "IE", "II"]
        J = [J_to_params(EE.J), J_to_params(EI.J), J_to_params(IE.J), J_to_params(II.J)]
        P = [P_to_params(EE.P), P_to_params(EI.P), P_to_params(IE.P), P_to_params(II.P)]
        w = [w_to_params(EE.w), w_to_params(EI.w), w_to_params(IE.w), w_to_params(II.w)]

        if EF is not None and IF is not None:
            self.feed_forward = True
            self.connections.append(EF)
            self.connection_names.append("EF")
            J.append(J_to_params(EF.J))
            P.append(P_to_params(EF.P))
            w.append(w_to_params(EF.w))
            self.connections.append(IF)
            self.connection_names.append("IF")
            J.append(J_to_params(IF.J))
            P.append(P_to_params(IF.P))
            w.append(w_to_params(IF.w))
            
            self.wg = WeightsGeneratorExact(J, P, w, EE.N + II.N, EF.N)
        else:
            self.wg = WeightsGeneratorExact(J, P, w, EE.N + II.N)
            self.feed_forward = False
        


    def check_bounds(self):
        if self.print_statement:
            print("Out of bounds found:\n\n")
        error = False
        error = self.kraynyukova_J_condition() or error
        error = self.kraynyukova_w_condition() or error
        error = self.W_tot_condition() or error
        error = self.efficacy_condition() or error
        error = self.connection_count_condition() or error
        if self.feed_forward:
            error = self.connection_count_condition_FF() or error
        print(self.wg.balance_in_ex_in())
        return error


    def kraynyukova_J_condition(self):
        error = False
        if self.feed_forward and not (self.EF.J / self.EF.root_N < self.IF.J / self.IF.root_N):
            if self.print_statement:
                print("kraynyukova_J_condition, J_EF/root(N_F) < J_IF/root(N_F)")
                print(self.EF.J / self.EF.root_N, self.IF.J / self.IF.root_N, '\n')
            error = True
        if self.feed_forward and not (self.IF.J / self.IF.root_N < self.EI.J / self.EI.root_N):
            if self.print_statement:
                print("kraynyukova_J_condition, J_IF/root(N_F) < J_EI/root(N_I)")
                print(self.IF.J / self.IF.root_N, self.EI.J / self.EI.root_N, '\n')
            error = True
        if not (self.EI.J / self.EI.root_N < self.EE.J / self.EI.root_N):
            if self.print_statement:
                print("kraynyukova_J_condition, J_EI/root(N_I) < J_EE/root(N_E)")
                print(self.EI.J / self.EI.root_N, self.EE.J / self.EI.root_N, '\n')
            error = True
        if not (self.EE.J / self.EE.root_N < self.II.J / self.II.root_N):
            if self.print_statement:
                print("kraynyukova_J_condition, J_EE/root(N_E) < J_II/root(N_I)")
                print(self.EE.J / self.EE.root_N, self.II.J / self.II.root_N, '\n')
            error = True
        if not (self.II.J / self.II.root_N < self.IE.J / self.IE.root_N):
            if self.print_statement:
                print("kraynyukova_J_condition, J_II/root(N_I) < J_IE/root(N_E)")
                print(self.II.J / self.II.root_N, self.IE.J / self.IE.root_N, '\n')
            error = True
        return error


    def kraynyukova_w_condition(self):
        error = False
        if not (self.EI.w < self.EE.w):
            if self.print_statement:
                print("kraynyukova_w_condition, w_EI < w_EE")
                print(self.EI.w, self.EE.w, '\n')
            error = True
        if abs(self.EE.w - self.II.w) > 10:  # 10 is a guess as in the paper they use the approx sign
            if self.print_statement:
                print("kraynyukova_w_condition, w_EE ~= w_II")
                print(self.EE.w, self.II.w, '\n')
            error = True
        if not (self.II.w < self.IE.w):
            if self.print_statement:
                print("kraynyukova_w_condition, w_II < w_IE")
                print(self.II.w, self.IE.w, '\n')
            error = True
        return error

    
    def W_tot_condition(self):
        error = False

        if self.feed_forward:
            upper_bound = self.EF.W_tot / self.IF.W_tot
        else:
            upper_bound = 1

        if not ((self.EE.W_tot / self.IE.W_tot) < (self.EI.W_tot / self.II.W_tot)):
            if self.print_statement:
                print("W_tot_condition, (W_tot_EE / W_tot_IE) < (W_tot_EI / W_tot_II)")
                print((self.EE.W_tot / self.IE.W_tot), (self.EI.W_tot / self.II.W_tot), '\n')
            error = True
        if not ((self.EI.W_tot / self.II.W_tot) < upper_bound):
            if self.print_statement:
                print("W_tot_condition, (W_tot_EI / W_tot_II) < (W_tot_EF / W_tot_IF)")
                print((self.EI.W_tot / self.II.W_tot), upper_bound, '\n')
            error = True
        return error

    def efficacy_condition(self):
        error = False
        for i, connection in enumerate(self.connections):
            if (connection.J / connection.root_N) < 0.2:
                if self.print_statement:
                    print(f"efficacy_condition, {self.connection_names[i]} is too low")
                    print(connection.J / connection.root_N, '\n')
                error = True

            if (connection.J / connection.root_N) > 2:
                if self.print_statement:
                    print(f"efficacy_condition, {self.connection_names[i]} is too high")
                    print(connection.J / connection.root_N, '\n')
                error = True
        return error


    def get_connection_count(self):
        E_total = 0
        I_total = 0
        weights = self.wg.generate_weight_matrix()
        for i in range(self.EE.N + self.II.N):
            row = np.array(weights[i].data)
            E_total += np.sum(self._normalise(row[:self.EE.N]))
            I_total += np.sum(self._normalise(np.abs(row[self.EE.N:])))
        return E_total, I_total


    def connection_count_condition(self):
        error = False
        for _ in range(self.test_trials):  # as weight matrix is random we will test it 10 times
            E_total, I_total = self.get_connection_count()

            if E_total / (self.EE.N + self.II.N) > 1200:
                if self.print_statement:
                    print("Too many E connections")
                    print(E_total / (self.EE.N + self.II.N), '\n')
                error = True
            elif E_total / (self.EE.N + self.II.N) < 350:
                if self.print_statement:
                    print("Too little E connections")
                    print(E_total / (self.EE.N + self.II.N), '\n')
                error = True

            if I_total / (self.EE.N + self.II.N) > 1200:
                if self.print_statement:
                    print("Too many I connections")
                    print(I_total / (self.EE.N + self.II.N), '\n')
                error = True
            elif I_total / (self.EE.N + self.II.N) < 350:
                if self.print_statement:
                    print("Too little I connections")
                    print(I_total / (self.EE.N + self.II.N), '\n')
                error = True

            if error:
                return error
        return error


    def get_connection_count_FF(self):
        connections_total = 0
        weights = self.wg.generate_feed_forward_weight_matrix()
        for i in range(0, self.EE.N + self.II.N):
            row = np.array(weights[i].data)
            connections_total += np.sum(self._normalise(row))
        return connections_total


    def connection_count_condition_FF(self):
        error = False
        for _ in range(self.test_trials):  # as weight matrix is random we will test it 10 times
            connections_total = self.get_connection_count_FF()

            if connections_total / (self.EE.N + self.II.N) > 120:
                if self.print_statement:
                    print("Too many feed-forward connections")
                    print(connections_total / (self.EE.N + self.II.N), '\n')
                error = True
            elif connections_total / (self.EE.N + self.II.N) < 35:
                if self.print_statement:
                    print("Too little feed-forward connections")
                    print(connections_total / (self.EE.N + self.II.N), '\n')
                error = True
            if error:
                return error
        return error


    @staticmethod
    def _normalise(array):
        return (array - np.min(array)) / (np.max(array) - np.min(array))


def _sigmoid(value, steepness=1, scaling=1):
    return scaling / (1 + np.exp(-steepness * value))


def _inverse_sigmoid(value, steepness=1, scaling=1):
    return - (1 / steepness) * np.log((scaling / value) - 1)


J_to_params = lambda x: _inverse_sigmoid(x, J_steep, J_scale)
P_to_params = lambda x: _inverse_sigmoid(x, P_steep, P_scale)
w_to_params = lambda x: _inverse_sigmoid(x, w_steep, w_scale)


params_to_J = lambda x: _sigmoid(x, J_steep, J_scale)
params_to_P = lambda x: _sigmoid(x, P_steep, P_scale)
params_to_w = lambda x: _sigmoid(x, w_steep, w_scale)


def mean_list_to_values(mean_list):
    return [params_to_J(mean_list[0]), params_to_J(mean_list[1]), params_to_J(mean_list[2]), params_to_J(mean_list[3]), 
            params_to_P(mean_list[4]), params_to_P(mean_list[5]), params_to_P(mean_list[6]), params_to_P(mean_list[7]),  
            params_to_w(mean_list[8]), params_to_w(mean_list[9]), params_to_w(mean_list[10]), params_to_w(mean_list[11])]


def noramalise_params_array(params_array):
    return [params_array[0]/J_scale, params_array[1]/J_scale, params_array[2]/J_scale, params_array[3]/J_scale, 
            params_array[4]/P_scale, params_array[5]/P_scale, params_array[6]/P_scale, params_array[7]/P_scale,  
            params_array[8]/w_scale, params_array[9]/w_scale, params_array[10]/w_scale, params_array[11]/w_scale]


def _get_random_params(std_vec):  # TODO: This function should draw valid parameters from constraint conditions
    return std_vec

def get_random_valid_params(trials=100, n=10000):
    std_vec = np.array([1/J_steep, 1/P_steep, 1/w_steep,
                        1/J_steep, 1/P_steep, 1/w_steep,
                        1/J_steep, 1/P_steep, 1/w_steep,
                        1/J_steep, 1/P_steep, 1/w_steep,])
    for _ in tqdm(range(trials)):
        sample = _get_random_params(std_vec) * np.random.randn(len(std_vec)) # TODO: Change this random to non-random
        EE = Connection(params_to_J(sample[0]), params_to_P(sample[1]), params_to_w(sample[2]), n)
        EI = Connection(params_to_J(sample[3]), params_to_P(sample[4]), params_to_w(sample[5]), n)
        IE = Connection(params_to_J(sample[6]), params_to_P(sample[7]), params_to_w(sample[8]), n)
        II = Connection(params_to_J(sample[9]), params_to_P(sample[10]), params_to_w(sample[11]), n)
        cc = ConstraintChecker(EE, EI, IE, II, print_statement=True, test_trials=1)
        error = cc.check_bounds()
        if not error:
            return EE, EI, IE, II
    return None, None, None, None


def print_values_in_code(EE: Connection, EI: Connection, IE: Connection, II: Connection, EF: Connection=None, IF:Connection=None):
    print("\n\nValues in code:\n")

    if EF is not None and IF is not None:
        print([EE.J, EI.J, IE.J, II.J, EF.J, IF.J, EE.P ,EI.P ,IE.P , II.P, EF.P, IF.P, EE.w, EI.w, IE.w, II.w, EF.w, IF.w], '\n')
        print("mean_list =", [J_to_params(EE.J), J_to_params(EI.J), J_to_params(IE.J), J_to_params(II.J), J_to_params(EF.J), J_to_params(IF.J),
                            P_to_params(EE.P), P_to_params(EI.P), P_to_params(IE.P), P_to_params(II.P), P_to_params(EF.P), P_to_params(IF.P),
                            w_to_params(EE.w), w_to_params(EI.w), w_to_params(IE.w), w_to_params(II.w), w_to_params(EF.w), w_to_params(IF.w),], '\n')
        
        J = [J_to_params(EE.J), J_to_params(EI.J), J_to_params(IE.J), J_to_params(II.J), J_to_params(EF.J), J_to_params(IF.J),]
        P = [P_to_params(EE.P), P_to_params(EI.P), P_to_params(IE.P), P_to_params(II.P), P_to_params(EF.P), P_to_params(IF.P),]
        w = [w_to_params(EE.w), w_to_params(EI.w), w_to_params(IE.w), w_to_params(II.w), w_to_params(EF.w), w_to_params(IF.w),]

        print("J_array =", J)
        print("P_array =", P)
        print("w_array =", w, '\n')

    else:
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

if __name__ == "__main__":

    SEARCH_INIT = bool(input("Search Params? (default False): "))

    N_E = 800
    N_I = 200
    N_F = 1000

    # N_E = 1600
    # N_I = 400
    # N_F = 1000

    # N_E = 2400
    # N_I = 600
    # N_F = 1000

    # N_E = 8000
    # N_I = 2000
    # N_F = 1000

    if SEARCH_INIT:
        EE, EI, IE, II = get_random_valid_params(trials=1000, n=N_E + N_I)
        if EE is None:
            print("Not found!")
            exit()
    
    else:
        INITIAL = bool(input("Initial Params? (default False): "))
        print("\nInitial: ", INITIAL, '\n')

        if INITIAL:
            # EE = Connection(35.78, 0.11, 32, N_E)  # 10000
            # EI = Connection(14.31, 0.45, 32, N_I)
            # IE = Connection(53.67, 0.11, 32, N_E)
            # II = Connection(17.89, 0.45, 32, N_I)

            # EE = Connection(19.60, 0.11, 32, N_E)  # 3000
            # EI = Connection(7.84, 0.45, 32, N_I)
            # IE = Connection(29.39, 0.11, 32, N_E)
            # II = Connection(9.80, 0.45, 32, N_I)

            # EE = Connection(16, 0.11, 32, N_E)  # 2000
            # EI = Connection(6.4, 0.45, 32, N_I)
            # IE = Connection(24, 0.11, 32, N_E)
            # II = Connection(8, 0.45, 32, N_I)

            # EE = Connection(11.31, 0.11, 32, N_E)  # 1000
            # EI = Connection(4.52, 0.45, 32, N_I)
            # IE = Connection(16.97, 0.11, 32, N_E)
            # II = Connection(5.66, 0.45, 32, N_I)

            EE = Connection(29.9035, 0.0335, 23.0645, N_E)  # 1000
            EI = Connection(87.2499, 0.8239, 5.527, N_I)
            IE = Connection(29.253, 0.9396, 40.9159, N_E)
            II = Connection(93.7406, 0.9263, 112.43, N_I)

            # EE = Connection(15, 0.22, 45, N_E)  # 1000
            # EI = Connection(7, 0.35, 45, N_I)
            # IE = Connection(20, 0.22, 45, N_E)
            # II = Connection(4, 0.35, 45, N_I)

            # EF = Connection(5, 0.11, 30, N_F)
            # IF = Connection(5, 0.11, 30, N_F)
            EF = None
            IF = None
        else:
            mean_list = [-5.6143,  4.2192, -2.7704,  2.3774, -2.6886, -3.6780,  1.6469,  5.2358, -3.0668, -1.6994, -1.6572,  0.9739]

            if len(mean_list) == 18:
                EE = Connection(params_to_J(mean_list[0]), params_to_P(mean_list[6]), params_to_w(mean_list[12]), N_E)
                EI = Connection(params_to_J(mean_list[1]), params_to_P(mean_list[7]), params_to_w(mean_list[13]), N_I)
                IE = Connection(params_to_J(mean_list[2]), params_to_P(mean_list[8]), params_to_w(mean_list[14]), N_E)
                II = Connection(params_to_J(mean_list[3]), params_to_P(mean_list[9]), params_to_w(mean_list[15]), N_I)
                EF = Connection(params_to_J(mean_list[4]), params_to_P(mean_list[10]), params_to_w(mean_list[16]), N_F)
                IF = Connection(params_to_J(mean_list[5]), params_to_P(mean_list[11]), params_to_w(mean_list[17]), N_F)

            elif len(mean_list) == 12:
                EE = Connection(params_to_J(mean_list[0]), params_to_P(mean_list[4]), params_to_w(mean_list[8]), N_E)
                EI = Connection(params_to_J(mean_list[1]), params_to_P(mean_list[5]), params_to_w(mean_list[9]), N_I)
                IE = Connection(params_to_J(mean_list[2]), params_to_P(mean_list[6]), params_to_w(mean_list[10]), N_E)
                II = Connection(params_to_J(mean_list[3]), params_to_P(mean_list[7]), params_to_w(mean_list[11]), N_I)
                EF = None
                IF = None
            else:
                raise ValueError(f"Expected an array of lenth 12 of 18 but found {len(mean_list)}.")

    cc = ConstraintChecker(EE, EI, IE, II, EF, IF)
    cc.check_bounds()
    print_values_in_code(EE, EI, IE, II, EF, IF)