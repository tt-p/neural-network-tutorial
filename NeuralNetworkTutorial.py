import random
from math import tanh, sqrt


class Connection(object):
    weight = 0
    delta_weight = 0

    def __init__(self, weight, delta_weight):
        self.weight = weight
        self.delta_weight = delta_weight


class Neuron(object):
    eta = 0.15
    alpha = 0.5

    index = 0
    output_val = 0
    gradient = 0
    output_weights = []

    def __init__(self, num_outputs, index):
        self.output_weights = [0 for k in range(num_outputs)]
        self.index = index

        for i in range(num_outputs):
            self.output_weights[i] = Connection(random.uniform(0, 1), 0)

    def feed_forward(self, prev_layer):
        sum = 0
        for i in range(len(prev_layer)):
            sum += prev_layer[i].output_val * prev_layer[i].output_weights[self.index].weight

        self.output_val = self.transfer_func(sum)

    def transfer_func(self, x):
        return tanh(x)

    def transfer_func_derivative(self, x):
        return 1.0 - x ** 2

    def calc_output_gradients(self, target_val):
        delta = target_val - self.output_val
        self.gradient = delta * self.transfer_func_derivative(self.output_val)

    def calc_hidden_gradients(self, next_layer):
        dow = self.sum_dow(next_layer)
        self.gradient = dow * self.transfer_func_derivative(self.output_val)

    def sum_dow(self, next_layer):
        sum = 0.0

        for i in range(len(next_layer) - 1):
            sum += self.output_weights[i].weight * next_layer[i].gradient

        return sum

    def update_weights(self, prev_layer):
        for i in range(len(prev_layer)):
            neuron = prev_layer[i]
            old_delta_weight = neuron.output_weights[self.index].delta_weight
            new_delta_weight = self.eta * neuron.output_val * self.gradient + self.alpha * old_delta_weight

            neuron.output_weights[self.index].delta_weight = new_delta_weight
            neuron.output_weights[self.index].weight += new_delta_weight


class Network(object):
    error = 0

    # recent_average_error
    rae = 0

    # recent_average_smoothing_factor
    rasf = 100.0

    def __init__(self, topology):
        self.layers = [0 for k in range(len(topology))]

        for i in range(0, len(self.layers)):

            self.layers[i] = [0 for k in range(topology[i] + 1)]

            for j in range(0, topology[i] + 1):
                self.layers[i][j] = Neuron(0 if i == len(topology) - 1 else topology[i + 1], j)

            self.layers[i][len(self.layers[i]) - 1].output_val = 1.0

    def feed_forward(self, input_values):
        for i in range(len(input_values)):
            self.layers[0][i].output_val = input_values[i]

        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i]) - 1):
                self.layers[i][j].feed_forward(self.layers[i - 1])

    def back_prop(self, target_values):
        self.error = 0

        output_layer = self.layers[len(self.layers) - 1]
        for i in range(len(output_layer) - 1):
            delta = target_values[i] - output_layer[i].output_val
            self.error += delta ** 2

        self.error /= sqrt(len(output_layer) - 1)
        self.rae = (self.rae * self.rasf + self.error) / (self.rasf + 1.0)

        for i in range(len(output_layer) - 1):
            output_layer[i].calc_output_gradients(target_values[i])

        for i in range(len(self.layers) - 2, 0, -1):
            hidden_layer = self.layers[i]
            next_layer = self.layers[i + 1]

            for j in range(len(hidden_layer)):
                hidden_layer[j].calc_hidden_gradients(next_layer)

        for i in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[i]
            prev_layer = self.layers[i - 1]

            for j in range(len(layer) - 1):
                layer[j].update_weights(prev_layer)

    def get_results(self, result_values):
        for i in range(len(self.layers[len(self.layers) - 1]) - 1):
            result_values.append(self.layers[len(self.layers) - 1][i].output_val)


f_and = lambda x, y: x and y

f_or = lambda x, y: x or y

f_xor = lambda x, y: (not x and y) or (x and not y)

f_nand = lambda x, y: not f_and(x, y)

f_nor = lambda x, y: not f_or(x, y)

f_xnor = lambda x, y: not f_xor(x, y)


def create_logic_data(n, f):
    data = {
        "input_values": [0 for k in range(n)],
        "target_values": [0 for k in range(n)],
        "result_values": [0 for k in range(n)],
    }
    for i in range(n):
        input1 = random.randint(0, 1)
        input2 = random.randint(0, 1)

        data["input_values"][i] = [input1, input2]
        data["target_values"][i] = [f(input1, input2)]
        data["result_values"][i] = []

    return data


def test(net, data_cerator, n_train, n_test, func, num_print=1):
    data_train = data_cerator(n_train, func)
    for i in range(n_train):
        input_values = data_train["input_values"][i]
        target_values = data_train["target_values"][i]
        result_values = data_train["result_values"][i]

        net.feed_forward(input_values)
        net.back_prop(target_values)
        net.get_results(result_values)

    data_test = data_cerator(n_test, func)
    mse = 0
    for i in range(n_test):
        input_values = data_test["input_values"][i]
        target_values = data_test["target_values"][i]
        result_values = data_test["result_values"][i]

        net.feed_forward(input_values)
        net.back_prop(target_values)
        net.get_results(result_values)

        mse += (int(target_values[0]) - result_values[0]) ** 2

        if i >= n_test - num_print:
            print(f"Inputs:{input_values} Target:[{int(target_values[0])}] Result:[{round(result_values[0], 2)}]")

    mse /= n_test

    print(f"Mean Square Error = {mse}\n")


if __name__ == "__main__":
    topology = [2, 4, 1]
    net = Network(topology)

    print("And Function")
    test(net, create_logic_data, 10000, 2000, f_and, 10)

    print("Or Function")
    test(net, create_logic_data, 10000, 2000, f_or, 10)

    print("Xor Function")
    test(net, create_logic_data, 10000, 2000, f_xor, 10)

    print("Nand Function")
    test(net, create_logic_data, 10000, 2000, f_nand, 10)

    print("Nor Function")
    test(net, create_logic_data, 10000, 2000, f_nor, 10)

    print("Xnor Function")
    test(net, create_logic_data, 10000, 2000, f_xnor, 10)
