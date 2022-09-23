from email import policy
import tensorflow as tf
import numpy as np
from DQNAgent import DQNAgent
import keras
from keras.losses import mean_squared_error
from keras.layers import Flatten
from keras.optimizers import Adam
from Env import DeepNav
#from DDPGAgent import MADDPGAgent
from replayBuffer import ReplayBuffer
from utils import flatten
from NNmodels import DDPGActor, DDPGCritic


class MADDPG:
    
    def __init__(self, env : DeepNav,  n_epochs=1000, n_episodes=10, tau=0.005,
                 gamma=0.99, l_r = 1e-5, bf_max_lenght=10000, bf_batch_size=64, path='./models/DDPG/'):
        
        self.env = env
        self.obs_space = self.env.getStateSpec()
        self.action_space = self.env.getActionSpec()
        
        
        self.n_agents = env.n_agents
        self.n_epochs = n_epochs
        self.n_episodes = n_episodes
        self.bf_max_lenght = bf_max_lenght
        self.batch_size = bf_batch_size
        self.path = path
        #self.agents = [MADDPGAgent(i, self.obs_space, self.action_space, gamma, l_r, tau)
                      # for i in range(self.n_agents)]
        self.agents = [
            {   
                'id' : agnt,
                'a_n' : DDPGActor(self.obs_space[1], self.action_space[1]),
                'target_a_n' : DDPGActor(self.obs_space[1], self.action_space[1]),
                'q_n' : DDPGCritic(self.obs_space[0] * self.obs_space[1], self.action_space[0]),
                'target_q_n' : DDPGCritic(self.obs_space[0] * self.obs_space[1], self.action_space[0]),
                'loss_fn' : mean_squared_error
            }
            for agnt in range(self.n_agents)
        ]
        self.rb = ReplayBuffer(env.getStateSpec(), env.getActionSpec(), self.n_agents,
                               self.bf_max_lenght, self.batch_size)
        self.gamma = gamma
        self.l_r = l_r
        
    def updateTarget(self):
        
            
        for i in self.agents:
            i['target_a_n'].set_weights(i['a_n'].weights)
            i['target_q_n'].set_weights(i['q_n'].weights) 

    def save(self):
        for i in self.agents:
            _id = i['id']
            i['q_n'].save(self.path + f'QNet_{_id}.h5')
            i['target_q_n'].save(self.path + f'QTargetNet_{_id}.h5')
        
            i['a_n'].save(self.path + f'ANet_{_id}.h5')
            i['target_a_n'].save(self.path + f'ATargetNet_{_id}.h5')
            
    def load(self):
        for i in self.agents:
            
            i['q_n'] = keras.models.load_model(self.path + f'QNet_{_id}.h5')
            i['target_q_n'] = keras.models.load_model(self.path + f'QTargetNet_{_id}.h5')

            i['a_n'] = keras.models.load_model(self.path + f'ANet_{_id}.h5')
            i['target_a_n'] = keras.models.load_model(self.path + f'ATargetNet_{_id}.h5')
            
    
    def normalize(self, a):
        norm = np.linalg.norm(a)
        
        return a * 1 / norm    
    
    
    def chooseAction(self, s : tf.Tensor, target : bool = False, training : bool = False):
        if s.ndim == 2:
            s = s.reshape((1, s.shape[0], s.shape[1]))
            
        actions = np.zeros((s.shape[0], self.n_agents, self.action_space[1]))
              
        
        for i in range(self.n_agents):
            x = s[:,i, :]
            #x = x.reshape((self.batch_size, 1, self.obs_space[1]))
            
            if not target:
            
                actions[:, i, :] = self.normalize(self.agents[i]['a_n'](x, training))    
            
            else:
                actions[:, i, :] = self.normalize(self.agents[i]['target_a_n'](x, training))
        return tf.convert_to_tensor(actions.squeeze())
    
    
    def policy(self, s):
        a = self.chooseAction(s)
        
        a += np.random.uniform(0, 1, a.shape)
        
        return a
        
        
    def Train(self):
        
        
        for i in range(self.n_epochs):
            for j in range(self.n_episodes):
                s = self.env.reset()
                reward = 0
                ts = 0
                H=500
                
                while 1:
                    
                    
                    a = self.policy(s)
                    s_1, r, done = self.env.step(a)
                    
                    self.rb.store(s, a, r, s_1, done)
                    
                    if self.rb.ready:
                        self._learn(self.rb.sample())
                        self.updateTarget()
                    s = s_1
                    ts +=1
                    
                    fmt = '*' * int(ts*10/H)
                    print(f'Epoch {i + 1} Episode {j + 1} |{fmt}| -> {ts}')
                    if done == 1 or ts > H:
                        print(f'Epoch {i + 1} Episode {j + 1} ended after {ts} timesteps')
                        ts=0
                        break
                    
                    
                if i % 10 == 0:
                    self.save()
                #print(f'Epoch: {i + 1} / {self.n_epochs} Episode {j + 1} / {self.n_episodes} Reward: {reward / ts}')        
                  
          
    def _learn(self, sampledBatch):
        s, a, r, s_1, dones = sampledBatch
        
        s = tf.convert_to_tensor(s)
        a = tf.convert_to_tensor(a)
        r = tf.convert_to_tensor(r)
        s_1 = tf.convert_to_tensor(s_1)
        dones = tf.convert_to_tensor(dones)
        
        
        for i in range(self.n_agents):
            s_agnt = s[:, i]
            a_agnt = a[:, i]
            r_agnt = r[:, i]
            s_1_agnt = s_1[:, i]
            dones_agnt = dones[i]
            agnt = self.agents[i]
            opt = Adam(self.l_r)
            with tf.GradientTape() as tape:
            
                #tape.watch(agnt['q_n'].trainable_variables)
                acts = self.chooseAction(s_1, True, True)
                acts = flatten(acts)
                s_1_ret = flatten(s_1)
                ret = self.gamma * agnt['target_q_n']([s_1_ret, acts], training=True)
                
                target = tf.convert_to_tensor(r_agnt + ret)
                
                
                    
                acts = flatten(a)
                obs = flatten(s)
                
                q_loss = agnt['loss_fn'](agnt['q_n']([obs, acts], training = True), target)
                
                
                
            q_grad = tape.gradient(q_loss, agnt['q_n'].trainable_variables)
            #print('QGRAD', q_grad)
            opt.apply_gradients(zip(q_grad, agnt['q_n'].trainable_variables))
            
            
            
            # Updating actors
            
            
            with tf.GradientTape(True) as tape:
                act = agnt['a_n'](s_agnt, training=True)
                
                acts = flatten(a)
                act = flatten(act)
                
                if i == 0:
                    
                    acts_ = tf.concat([
                    act,
                    acts[:, 1:-1]
                    ], axis=1)
                elif 0 < i < self.n_agents -1:
                    
                    acts_ = tf.concat([
                    acts[:, 0: 2*i],
                    act,
                    acts[:, (2*i) + 1:-1]
                    ], axis=1)
                
                else:
                    
                    acts_ = tf.concat([
                    acts[:, 0: 2*(self.n_agents)-2],
                    act,
                    
                    ], axis=1)
                
                
                
                q_values = agnt['q_n']([obs, acts_],training=True)
                loss = tf.math.reduce_mean(q_values)
                    
                    
                    
                
                
                
            actorGrad = tape.gradient(loss, agnt['a_n'].trainable_variables)
            opt.apply_gradients(zip(actorGrad, agnt['a_n'].trainable_variables))
            
                              
            
            
        
        
if __name__ == '__main__':
    
     env = DeepNav(3, 0)
     p = MADDPG(env)
     
     p.Train()   
