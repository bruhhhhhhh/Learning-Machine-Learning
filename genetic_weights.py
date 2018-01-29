#-----------------------------------Genetic algorithm class
import numpy as np
import random

class weightvalues:
    def __init__(self, nvar):
        self.nvar = nvar
        self.values = np.random.uniform(-1,1,nvar)
        self.fitness = 0

    def set_fitness(self, fitness):        
        self.fitness = fitness

    def mutate(self, noise):
        self.values =  self.values + noise*np.random.uniform(-1,1,self.nvar)
        for x in range(len(self.values)):
            self.values[x] = np.tanh(self.values[x])

    def breed(self, values2):
        childgene = np.array([])
        values1 = self.values
        values2 = values2.values
        for x in range(len(values1)):
            if random.randint(0,2)==0:
                childgene = np.append(childgene, values1[x])
            else:
                childgene = np.append(childgene, values2[x])
        return childgene
    
    def getvalues(self):
        return self.values
