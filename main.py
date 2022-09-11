from network import generate_net, print_net,  forward, brackprop, update_net
from dataset import generate_dataset, format_output


def main():

    l_rate = 0.01

    #first i generate a dataset
    data = generate_dataset(100000)

    #we create a net with number of inputs, hidden layers and outputs
    net = generate_net(4, [4], 2)
    print_net(net)

    #training
    for sample in data:
        #forward propagation
        forward(net, sample[0])
        #backpropagation, here we calculate the deltas
        brackprop(net, sample[1])

        #gradient descend, here we change de weigths and bias to get closer
        #to the correct criteria of avaluation
        update_net(net, sample[0], l_rate)

    #new dataset to avoid overfiting
    data = generate_dataset(100)

    oks = 0
    iteratios = 10000


    #in that iteration we start with our prediction task
    for it in range(iteratios):
        for test_sample in data:
            forward(net, test_sample[0])
            output = format_output([ neuron['a'] for neuron in net[len(net)-1] ])
            #the results
            print(f"expected: {test_sample[1]} given: {output}")
            if output == test_sample[1]:
                oks += 1
    #the acurracy
    print(f"{((oks / len(data)/100))} % of accuracy")







if __name__ == '__main__':
    main()