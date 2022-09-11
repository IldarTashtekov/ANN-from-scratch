from math import exp
from random import random

def print_net(net):
    print(f"Net with {len(net)} layers (excluding input)")
    print(f"{len(net[0][0]['weights'])} inputs : layer L-{len(net)}")
    for i, hidden in enumerate(net[:-1]):
        print(f"{len(hidden)} neurons in hidden : layer L-{len(net)-1-i}")
    print(f"{len(net[-1])} neurons in output : layer L")


def generate_net(n_inputs,hidden_layers,n_outputs):
    net=list()


    #hidden layers
    for n_neurons in hidden_layers:
        #for every neuron in hidden layer we create a neuron with num of weigths equal to n_umputs
        net.append([create_neuron(n_inputs) for i in range(n_neurons)])

        n_inputs=n_neurons

    #for every neuron in output layer we create a neuron with num of weigths equal to n_neurons
    net.append([create_neuron(n_neurons) for i in range(n_outputs)])

    return net


#random value between -1 and 1
def rand_value():
    return random() * 2.0 - 1.0


def create_neuron(n_weights):
    return{
        'weights': [rand_value() for i in range(n_weights)],
        'bias': rand_value(),
        'z': 0.0,
        'a': 0.0,
        'w_part_deriv': [0.0 for i in range(n_weights)],
        'bias_part_deriv': 0.0
    }


def perceptorn(weights, inputs, bias):
    # y = w1 i1 + w2 i2 + ... + wn in + b
    return sum([weights[i] * inputs[i] for i in range(len(weights))]) + bias

#our activation function is a sigmoid function
def sigmoid(z):
    return 1.0 / (1.0 + exp(-z))

# A(Z(w,a,b))
def percpetron_output(z):
    return sigmoid(z)



def forward(net, inputs):

    current_input = inputs
    # for every layer in net
    for layer in net:
        #list with inputs of the actual layer
        next_input = list()
        #for every neuron in layer
        for neuron in layer:
            #first calculate z(w,i,b)= w1 i1 + w2 i2 + ... + wn in + b
            neuron['z'] = perceptorn(neuron['weights'], current_input, neuron['bias'])
            #then calculate a(z)=sigmoid(z)
            neuron['a'] = percpetron_output(neuron['z'])
            #add the layer a outputs to next input array
            next_input.append(neuron['a'])
        current_input = next_input

    #the last outputs
    return current_input


def sigmoid_derivative(z):
    output = sigmoid(z)
    return output * (1.0 - output)

def calculate_deltas(net, y):

    output_layer_index = len(net) - 1

    #backward iteration
    for i_layer in reversed(range(len(net))):
        #actual layer
        layer = net[i_layer]

        for i_neuron, neuron in enumerate(layer):
            #if layer is output layer our delta is neuron derivative * cost function derivative
            if i_layer == output_layer_index:
                neuron['delta'] = sigmoid_derivative(neuron['z']) * (neuron['a'] - y[i_neuron])
            else:
            #if layer is not output layer delta calculation is, is like average : sum ( neuronWi * deltaLayer i+1) * 1/n
                delta_L_plus_1 = sum([next_neuron['weights'][i_neuron] * next_neuron['delta'] for next_neuron in net[i_layer+1]])
                # normalizar resultado :
                delta_L_plus_1 /= len(net[i_layer+1])
                #assign delta to neurons of layer
                neuron['delta'] = sigmoid_derivative(neuron['z']) * delta_L_plus_1



def update_net(net, input_row, l_rate):

    inputs = input_row[:]
    for layer in net:
        for neuron in layer:
            l_delta = l_rate * neuron['delta'];
            for i_w in range(len(neuron['weights'])):
                #gradient descend for every neuron neuronW= neuronW - lr * delta
                neuron['weights'][i_w] -= l_delta * inputs[i_w]
            #same with bias
            neuron['bias'] -= l_delta


def brackprop(net, y):
    calculate_deltas(net, y)
