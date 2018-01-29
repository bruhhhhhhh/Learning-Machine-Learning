from neuralnet_v2 import fully_interconnected_net
from genetic_weights import *
import gym, random, time

env = gym.make('LunarLander-v2')
env.reset()

print('Action Space: ', env.action_space)
print('Observation Space', env.observation_space)

def calc_fitness(net, env):
    lastobs = env.reset()
    points = 0
    while True:
        out = net.go(lastobs)
        highest = 0
        action = 0
        for x in range(out.size):
            if out[x]>highest:
                highest = out[x]
                action = x
                
        obs, reward, done, info = env.step(action)
        
        if done:
            break
        
        points+=reward
        lastobs = obs
    return points

def watchrun(net, env):
    lastobs = env.reset()
    points = 0
    while True:
        env.render()
        out = net.go(lastobs)
        highest = 0
        action = 0
        for x in range(out.size):
            if out[x]>highest:
                highest = out[x]
                action = x
                
        obs, reward, done, info = env.step(action)
        
        if done:
            break

        points+=reward
        lastobs = obs
        time.sleep(0.01)
    return points 

#--------------------------neural net creation
noinputs = 8
no_first_layer_neurons = 4
no_second_layer_neurons = 4

#input layer
mynet = fully_interconnected_net(noinputs)
#first layer
mynet.addlayer(no_first_layer_neurons)
#second layer
mynet.addlayer(no_second_layer_neurons)

#----------------------------------------------------Created neural net

#initialises required variables
time_since_mutation = 0
current_gen = []

#hyper-parameters
gen_size = 250
no_gens = 100
initial_mutation_rate = 15
max_fitness = 40
noise = 0.2
per_unit_replace = 0.95 #percentage of population to replace per generation

#----------------------------------------------------------Main program
print('Number of synapses: ', mynet.no_synapses)
for x in range(gen_size):
    weights = weightvalues(mynet.no_synapses)
    current_gen.append(weights)

#uncomment to watch pre-trained model
'''
optimized_weights = np.array([ 0.9923431,   0.46266229,  0.18109883, -0.77854646, -0.82722733,  0.8878298,
  0.51047041,  0.40009965,  0.67029461,  0.21684,    -0.34132481, -0.61918927,
 -0.19732634, -0.65385651,  0.15373509, -0.38985687, -0.68981624, -0.37286401,
 -0.95971062,  0.62724917, -0.81693156, -0.93943943,  0.28185905, -0.03543955,
  0.02355076, -0.15251006, -0.03258814,  0.64407134, -0.18283883, -0.89261078,
 -0.51102237, -0.65221105,  0.29822377, -0.82827524, -0.7685054,   0.4913199,
 -0.80890519,  0.94580548,  0.8214,      0.57605382,  0.63841679, -0.29016335,
  0.43346327,  0.6969483,  -0.10500797,  0.19041321,  0.04133948,  0.58425484])
mynet.set_all_weights(optimized_weights)
for _ in range(30):
    print('Fitness ', watchrun(mynet, env))
'''

for gen_no in range(no_gens):
    
    fitnesses = [] #used to store fitness of population
    unique_values = [] #used to calculate diversity of population

    #for each set of weights, test fitness
    for n in range(gen_size):
        test_weights = current_gen[n].getvalues()
        for weight in test_weights:
            if weight not in unique_values:
                unique_values.append(weight)
        #setting the weights to be tested
        mynet.set_all_weights(test_weights)

        #watch some runs every 10 generations
        if gen_no%10==0:
            if n>=round(gen_size*per_unit_replace):
                current_gen[n].set_fitness(watchrun(mynet, env))
        current_gen[n].set_fitness(calc_fitness(mynet, env))

    #sort generation by fitness        
    current_gen.sort(key=lambda x:x.fitness)

    diversity = len(unique_values)/(mynet.no_synapses*gen_size)
    noise = 1-diversity
    if noise>0.5:
        noise = 0.5
    if diversity<0.3:
        mutation_rate = initial_mutation_rate+10
        noise = 0.3
    else:
        mutation_rate = initial_mutation_rate
    
    for x in range(round(gen_size*per_unit_replace), gen_size):
        print('Weights: ', current_gen[x].getvalues(),'\n ',current_gen[x].fitness)

    fitnesses = [current_gen[x].fitness for x in range(round(gen_size*per_unit_replace), gen_size)]
    avg_fitness = np.mean(fitnesses)

    print('\n','Generation no: ',gen_no, '-')
    print('\nAverage Fitness of top' ,str((100-per_unit_replace*100)),'%: ', avg_fitness)
    print('Mutation Rate:  ',mutation_rate)
    print('Diversity: ', diversity)
    print('Noise: ', noise)
    if gen_no!=no_gens-1:
        #breeds generation
        for x in range(round(gen_size*0.7)):
            par1 = random.randint(round(gen_size*per_unit_replace), gen_size-1)
            par2 = random.randint(round(gen_size*per_unit_replace), gen_size-1)
            while par2==par1:
                par2 = random.randint(round(gen_size*per_unit_replace), gen_size-1)
                
            current_gen[x].values = current_gen[par1].breed(current_gen[par2])
        #mutates generation
        for item in current_gen:    
            time_since_mutation += mutation_rate
            if time_since_mutation>=50:
                item.mutate(noise)
                time_since_mutation = 0

#runs with best weights
'''test_weights = current_gen[gen_size-1].getvalues()
mynet.set_all_weights(test_weights)

for x in range(100):
    print('Fitness ', watchrun(mynet, env))
'''
        
env.close()
            
