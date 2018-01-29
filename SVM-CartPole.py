import gym, random
from gym import wrappers
from sklearn.svm import SVC
import numpy as np

env = gym.make('CartPole-v1')
env = wrappers.Monitor(env, '/home/haroon/Desktop/Machine Learning/cartpole-experiment-2', force=True)

#plays many games, trying random moves and learns a policy based on the outcomes

clf = SVC()

#x_train stores the observations
x_train = np.array([])
#y_train stores the move taken
y_train = np.array([])

last_ob = env.reset()
for x in range(10000):
        move = random.randint(0,1)
        observation, reward, done, info = env.step(move)
            
        if done:
            last_ob = env.reset()
        elif abs(observation[2])<=abs(last_ob[2]) or abs(observation[3])<=abs(last_ob[3]) and abs(observation[0])<=abs(last_ob[0]):
            x_train = np.append(x_train, last_ob)
            y_train = np.append(y_train, move)
            last_ob = observation
        else:
            last_ob = observation

x_train = x_train.reshape(-1,4)
clf.fit(x_train, y_train)

for x in range(50):
    
    for x in range(10000):
        move = clf.predict(np.array(last_ob).reshape(1,-1))
        observation, reward, done, info = env.step(int(move[0]))
                    
        if done:
            last_ob = env.reset()
        else:
            last_ob = observation
    
    print('Training --------------------')
