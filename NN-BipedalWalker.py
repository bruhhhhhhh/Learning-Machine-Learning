from neuralnet_v2 import fully_interconnected_net
from genetic_weights import *
import gym, random, time

env = gym.make('BipedalWalker-v2')
env.reset()

print('Action Space: ', env.action_space)
print('Observation Space', env.observation_space)


#runs simultation to calculate fitness
def calc_fitness(net, env):
    lastobs = env.reset()
    points = 0
    while True:
        out = net.go(lastobs)                
        obs, reward, done, info = env.step(out)
        
        if done:
            break

        points+=reward
        lastobs = obs
    return points

#runs and shows simulation
def watchrun(net, env):
    lastobs = env.reset()
    points = 0
    while True:
        out = net.go(lastobs)
        env.render()
        obs, reward, done, info = env.step(out)
        
        if done:
            break

        points+=reward
        lastobs = obs
        time.sleep(0.01)
    return points
    


#--------------------------neural net creation
noinputs = 24
no_first_layer_neurons = 4
no_second_layer_neurons = 4
no_third_layer_neurons = 4

#input layer
mynet = fully_interconnected_net(noinputs)
#first layer
mynet.addlayer(no_first_layer_neurons)
#second layer
mynet.addlayer(no_second_layer_neurons)
#third layer
mynet.addlayer(no_third_layer_neurons)

#----------------------------------------------------Created neural net

#initialises required variables
time_since_mutation = 0
current_gen = []

#hyper-parameters
gen_size = 300
no_gens = 150
initial_mutation_rate = 20
noise = 0.2
per_unit_replace = 0.9 #percentage of population to replace per generation

#----------------------------------------------------------Main program
print('Number of synapses: ', mynet.no_synapses)
for x in range(gen_size):
    weights = weightvalues(mynet.no_synapses)
    current_gen.append(weights)

#uncomment to watch pre-trained model
'''
optimized_weights = np.array([ 0.54670736, -0.29002195, -0.54269303,  0.71206659, -0.52373633, -0.18898345,
 -0.2929876 ,  0.04590431, -0.69437918,  0.1950015 ,  0.32554398, -0.74241738,
  0.4967644 , -0.44107566, -0.82438315,  0.28770777, -0.71327288,  0.06274322,
 -0.5046393 ,  0.09521248, -0.46801074, -0.5919123 , -0.31847738, -0.73380369,
 -0.09483436,  0.77989737, -0.12585044,  0.76168812, -0.25097859, -0.03428696,
 -0.29995409, -0.49723959,  0.20344617,  0.57728245, -0.44568008, -0.6515595,
  0.15134735, -0.19304158, -0.64901743,  0.05998842, -0.45397085, -0.27985297,
 -0.21896456, -0.41399423, -0.44299203, -0.10559664,  0.03065603, -0.60155784,
 -0.17191649,  0.26812571, -0.50146736,  0.29728699, -0.77665615, -0.83109932,
 -0.36396538,  0.70583317, -0.19719408,  0.43563501, -0.6664113 ,  0.14419504,
  0.31536418,  0.65717455, -0.63532492, -0.39848343,  0.4904293 ,  0.3623951,
  0.77176451,  0.03773144, -0.15580438,  0.68177909, -0.63047226,  0.13627389,
 -0.36814666,  0.50657352,  0.26363799,  0.54191208,  0.68192652,  0.02026647,
  0.65609572, -0.4063403 , -0.29007338,  0.14688142, -0.5093765 ,  0.70515058,
  0.16422842, -0.31464798, -0.30343923,  0.55778111,  0.24599404, -0.59930698,
  0.1778805 ,  0.44833184, -0.33382641, -0.57066179,  0.25387601,  0.28341742,
 -0.08348564, -0.15479878,  0.61784773, -0.5016239 , -0.76730235,  0.83529334,
  0.13823586, -0.08229324,  0.14501365,  0.51736414,  0.48821622,  0.20341164,
  0.36366231,  0.01773771,  0.08901062,  0.60412175, -0.45992462,  0.75553599,
 -0.57243547,  0.50897093, -0.53062448, -0.34414859,  0.35100765,  0.47895217,
  0.12115113, -0.16504049, -0.14553303, -0.24958458,  0.53953707, -0.28416409,
  0.35731535, -0.02420381])
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
        if gen_no%10==0 and gen_no!=0:
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
        mutation_rate = initial_mutation_rate+5
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
        for x in range(round(gen_size*per_unit_replace)):
            par1 = random.randint(round(gen_size*per_unit_replace), gen_size-1)
            par2 = random.randint(round(gen_size*per_unit_replace), gen_size-1)
            while par2==par1:
                par2 = random.randint(round(gen_size*per_unit_replace), gen_size-1)    
            current_gen[x].values = current_gen[par1].breed(current_gen[par2])

            #mutates generation    
            time_since_mutation += mutation_rate
            if time_since_mutation>=50:
                current_gen[x].mutate(noise)
                time_since_mutation = 0

#runs with best weights
test_weights = current_gen[gen_size-1].getvalues()
mynet.set_all_weights(test_weights)

for x in range(100):
    print('Fitness ', watchrun(mynet, env))


env.close()
            
