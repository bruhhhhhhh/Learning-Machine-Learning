import numpy as np

class fully_interconnected_net:
    def __init__(self,noinputs, act_func='tanh'):
        #initialising layers list
        self.layers = []
        self.synapse_layers = []
        self.no_synapses = 0
    
        #adding input layer
        self.layers.append(np.zeros(noinputs))
        
        #defining choices of activation function
        def tanh(x, derivative=False):
            if derivative:
                return 1-(x**2)
            return np.tanh(x)
        def sigmoid(x, derivative=False):
            if derivative:
                return x*(1-x)
            return 1/(1+np.exp(-x))

        #setting activation function
        if act_func=='tanh':
            self.act_func=tanh
        elif act_func=='sigmoid':
            self.act_func = sigmoid
        
    def go(self,inputs):
	#sets values for input layer
        self.layers[0]=inputs
        #carries out operations
        for x in range(len(self.synapse_layers)):

            self.layers[x+1] = self.act_func(np.matmul(self.layers[x], self.synapse_layers[x]))
            
        return self.layers[-1]     
    
    def addlayer(self, noneurons):
        self.layers.append(np.zeros(noneurons))        
        arr = np.random.uniform(-1, 1, (len(self.layers[-2]), len(self.layers[-1])))
        self.synapse_layers.append(arr)
        self.no_synapses += len(self.layers[-1]) * len(self.layers[-2])


    def set_synapse_weights(self, synapse_index, weights):
        self.synapse_layers[synapse_index] = weights

    def set_all_weights(self, weights):
        ind = 0
        for x in range(len(self.synapse_layers)):
            self.synapse_layers[x] = weights[ind:ind+self.synapse_layers[x].size].reshape(len(self.layers[x]), len(self.layers[x+1]))
