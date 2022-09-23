import tensorflow as tf
import numpy as np

from DQNAgent import DQNAgent
from ORCAAgent import ORCAAgent
from Env import DeepNav
from replayBuffer import ReplayBuffer

class MADQNMix:
    
    def __init__(self, env : DeepNav, n_smarties : int,target_update=1000, n_epochs= 1000, n_episodes=100, gamma=0.9, path='./models/DQN/',epsilon=1.0, buffer_max_length=10000, buffer_batch_size=64) -> None:
        
        self.env = env
        self.input_shape = env.getStateSpec()[1]
        self.output_shape = env.getActionSpec()[1]
        self.n_agents = self.env.n_agents
        self.n_epochs = n_epochs
        self.n_episodes = n_episodes
        self.epsilon = epsilon
        self.smarties = np.random.randint(0, self.n_agents, n_smarties).tolist()
        self.rb = ReplayBuffer(self.env.getStateSpec(), self.env.getActionSpec(), self.n_agents, buffer_max_length, buffer_batch_size)
        self.target_update = target_update
        self.agents = []
        for i in range(self.n_agents):
            if i in self.smarties:
                self.agents.append(DQNAgent(i, self.input_shape, self.output_shape))
                
            else :
                self.agents.append(ORCAAgent(i, env))
        
    def chooseAction(self, states : tf.Tensor):
        actions = np.zeros((self.n_agents, self.output_shape))
        
        for i in range(self.n_agents):
            actions[i] = self.agents[i].act(states[:, i])
            
        return actions
    
    def updateAgents(self):
        
        for i in self.smarties:
            self.agents[i].updateTarget()
    
    def save(self):
        for i in self.smarties:
            
            i.saveModel()
            
    def load(self):
        for i in self.smarties:
            i.loadModel()
            
    def epsilonGreedy(self, s):
        
        if np.random.rand() < self.epsilon:
            return np.random.uniform(0, 1, (self.n_agents, self.output_shape))
        
        return self.chooseAction(s)
        
    def Train(self):
        
        print(self.n_epochs)
        for i in range(self.n_epochs):
            for j in range(self.n_episodes):
                s = self.env.reset()
                reward = 0
                ts = 0
                while 1:
                    a = self.epsilonGreedy(s)
                    s_1, r, done = self.env.step(a)
                    
                    self.rb.store(s, a, r, s_1, done)
                    
                    if self.rb.ready:
                        self.learn(self.rb.sample())
                    
                    s = s_1
                    ts +=1
                    
                    if ts % self.target_update == 0:
                        self.updateAgents()
                    
                    if done == 1:
                        break
                    
                    
                self.epsilon -= 1 / (self.n_epochs * self.n_episodes)
                self.save()
                print(f'Epoch: {i + 1} / {self.n_epochs} Episode {j + 1} / {self.n_episodes} Reward')        
    
    def learn(self, sampled_batch):
            s, a, r, s_1, dones = sampled_batch
            
            for i in self.smarties:
                batch = [s[:, i], a[:, i], r[:, i], s_1[:, i], dones]
                self.agents[i].train(batch)


if __name__ == '__main__':
    
     env = DeepNav(10, 0)
     p = MADQNMix(env, 2)
     
     p.Train()   
     
     