__author__ = 'Xin Yang'

# Back-Propagation Neural Networks
#
import sys
import math
import random
import pandas as pd
import numpy as np


# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return y*(1-y)

class NN:
    def __init__(self, ni, nh1, nh2, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh1 = nh1
        self.nh2 = nh2
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah1 = [1.0]*self.nh1
        self.ah2 = [1.0]*self.nh2
        self.ao = [1.0]*self.no

        # create weights
        self.wi = makeMatrix(self.ni, self.nh1)
        self.wo = makeMatrix(self.nh2, self.no)
        self.wh = makeMatrix(self.nh1, self.nh2)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh1):
                self.wi[i][j] = rand(-0.5, 0.5)

        for j in range(self.nh2):
            for k in range(self.no):
                self.wo[j][k] = rand(-0.5, 0.5)

        for j in range(self.nh1):
            for k in range(self.nh2):
                self.wh[j][k] = rand(-0.5, 0.5)

        # last change in weights for momentum
        self.ci = makeMatrix(self.ni, self.nh1)
        self.co = makeMatrix(self.nh2, self.no)
        self.ch = makeMatrix(self.nh1, self.nh2)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden 1 activations
        for j in range(self.nh1):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah1[j] = sigmoid(sum)

        # hidden 2 activations
        for j in range(self.nh2):
            sum = 0.0
            for i in range(self.nh1):
                sum = sum + self.ah1[i] * self.wh[i][j]
            self.ah2[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh2):
                sum = sum + self.ah2[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden2
        hidden2_deltas = [0.0] * self.nh2
        for j in range(self.nh2):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden2_deltas[j] = dsigmoid(self.ah2[j]) * error

        # calculate error terms for hidden1
        hidden1_deltas = [0.0] * self.nh1
        for j in range(self.nh1):
            error = 0.0
            for k in range(self.nh2):
                error = error + hidden2_deltas[j] * self.wh[j][k]
            hidden1_deltas[j] = dsigmoid(self.ah1[j]) * error


        # update output weights
        for j in range(self.nh2):
            for k in range(self.no):
                change = output_deltas[k]*self.ah2[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh1):
                change = hidden1_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(sigmoid(targets[k])-self.ao[k])**2
        return error



    def train(self, data, iterations, N, M):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in data:
                inputs = p[:-1]
                targets = [p[-1]]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)

        print('layer 0 (input layer):')
        for x in range(len(self.wi)-1):
            print('Neuron', x+1 ,'weight:' , self.wi[x])

        print('layer 1 (hidden layer 1):')
        for y in range(len(self.wh)):
            print('Neuron', y+1, 'weight:', self.wh[y])

        print('final layer (final hidden layer):')
        for z in range(len(self.wo)):
            print('Neuron', z+1, 'weight:', self.wo[z])

        print('the total training error is %-.5f' % error)


    def test(self, data):
        error = 0.0
        for p in data:
            inputs = p[:-1]
            targets = [p[-1]]
            outputs = self.update(inputs)
            error = error + 0.5 * (sigmoid(targets[0]) - outputs[0]) ** 2
        print ('the total test error is %-.5f' % error)




def demo():

    #ask the user to initial the NN
    # python XinNN.py car_clean.dat 80  2 3 0.5 0.1 100
    # 80 -> train percentage
    # 2  -> hidden_layer 1
    # 3  -> hidden_layer 2
    # 0.5 ->learning rate
    # 0.1 ->momentum factor
    # 100 -> iterations

    # Initialize parameters
    filename = sys.argv[1]
    trainp = int(sys.argv[2])
    hlayer1 = int(sys.argv[3])
    hlayer2 = int(sys.argv[4])
    lrate = float(sys.argv[5])
    mfactor = float(sys.argv[6])
    iterations = int(sys.argv[7])

    data = pd.read_csv(filename, delimiter=",", header=0).as_matrix()

    #split train, test set
    num_data = int(len(data) * trainp / 100)
    train = data[0:num_data]
    test = data[num_data:len(data)]

    # create a network with several inputs, two hidden layers, and one output node
    n = NN(len(data[0])-1, hlayer1, hlayer2, 1)
    # train it with some patterns
    n.train(train,iterations,lrate,mfactor)
    # test it
    n.test(test)



if __name__ == '__main__':
    demo()