from pyexpat import model
import tensorflow as tf
import keras
import numpy as np
from NNmodels import qNetFC
from keras.losses import mean_squared_error




class DQNAgent:
    
    def __init__(self, agnt_id, input_shape, output_shape, gamma=0.9, learning_rate=1e-04, path='./models/DQN/') -> None:
        
        self.id = agnt_id
        self.state_space = input_shape
        self.action_space = output_shape
        
        self.q_net = qNetFC(input_shape=input_shape, output_shape=output_shape)
        self.q_target_net = qNetFC(input_shape=input_shape, output_shape=output_shape)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.path = path
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self.loss_fn = mean_squared_error
        
    def updateTarget(self):
        self.q_target_net.set_weights(self.q_net.get_weights())
        
    def saveModel(self):
        self.q_net.save(self.path + f'QNet_{self.id}.h5')
        self.q_target_net.save(self.path + f'QTargetNet_{self.id}.h5')
        
    def loadModel(self):
        self.q_net = keras.models.load_model(self.path + f'QNet_{self.id}.h5')
        self.q_target_net = keras.models.load_model(self.path + f'QTargetNet_{self.id}.h5')
        
    def normalize(self, a):
        norm = np.linalg.norm(a)
        
        return a * 1/ norm 
    
    def act(self, state : tf.Tensor):
        return self.normalize(self.q_net(state, training=False).numpy())
        
    
    def train(self, sampled_batch):
        s, a, r, s_1, dones = sampled_batch
        s = tf.convert_to_tensor(s)
        a = tf.convert_to_tensor(a)
        r = tf.convert_to_tensor(r)
        s_1 = tf.convert_to_tensor(s_1)
        dones = tf.convert_to_tensor(dones)
        
        f_r = self.q_target_net.predict(s_1)

        
        updated_q_values = r + self.gamma * tf.reduce_max(f_r, axis=1)
        updated_q_values = updated_q_values * (1 - dones) - dones
        #print(updated_q_values.shape)
        
        mask = tf.one_hot(a.shape[0], self.action_space)
        
        
        with tf.GradientTape() as tape:
            q_values = self.q_net(s)
            q_action = tf.reduce_sum(tf.multiply(q_values, mask), axis=1)
            
            loss = self.loss_fn(updated_q_values, q_action)
            
        grads =  tape.gradient(loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))
        
        
                    
        
        
        
        
        
    
if __name__ == '__main__':
    
    dqn = DQNAgent(0,3, 2)
    #print(dqn.act(np.random.uniform(0, 1, (10, 3))))
    #print(dqn.q_net.summary())
    
    
    
    
    
         