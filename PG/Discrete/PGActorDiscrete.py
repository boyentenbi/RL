import tensorflow as tf
import tflearn
import numpy as np

class PGActorDiscrete(object):

    def __init__(self, sess, state_dim, action_dim,  actor_learning_rate, critic_learning_rate):  
        
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        
        # Create the action graph
        self.state, self.probs = self.create_actor_network()
        self.network_params = tf.trainable_variables()
        
        # Create the loss calculation and optimizer graph
        self.old_actions, self.old_advantages, self.loss = self.create_loss()
        self.optimizer = tf.train.AdamOptimizer(self.actor_learning_rate)
        self.optimize = self.optimizer.minimize(self.loss, var_list = self.network_params)
        
        # Create value graph
        self.value = self.create_value_network()
        # Create value loss calculation and optimizer graph
        self.old_hindsight_values, self.value_loss= self.create_value_loss()
        self.optimize_value = tf.train.AdamOptimizer(self.critic_learning_rate).minimize(self.value_loss)

    def create_actor_network(self):
        # state placeholders (these could be sequences for recurrent model)
        state = tflearn.input_data(shape=[None, self.s_dim])
        # feedforward / recurrent model to action 
        
        l1 = tflearn.fully_connected(state, 30, activation='relu')
        l2 = tflearn.fully_connected(l1, 10, activation='relu')
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        #unscaled_means = tflearn.fully_connected(l2, self.a_dim, activation='tanh', weights_init=w_init)
        #unscaled_stds = tflearn.fully_connected(l2, self.a_dim, activation ='sigmoid', weights_init = w_init)
                                         
        #stds = tf.mul(unscaled_stds, self.action_bound/8.)
        #means = tf.mul(unscaled_means, self.action_bound) # Scale output to -action_bound to action_bound
        logits = tflearn.fully_connected(l2, 2)
        probs = tf.nn.softmax(logits)
        return state, probs
    
    def create_loss(self):
        #old_states = tflearn.input_data(shape=[None, self.s_dim])
        old_actions = tflearn.input_data(shape=[None, self.a_dim])
        old_advantages = tflearn.input_data(shape=[None, 1])
        
        #vars = self.stds**2
        #probs = tf.exp(-((old_actions-self.means)**2)/(2*vars))/tf.sqrt(2*vars * np.pi)
        multiplier = tf.concat(1, [(1-old_actions), old_actions])
        used_probs = tf.reduce_sum(self.probs * multiplier, reduction_indices = 1, keep_dims = True)
        #used_probs = tf.Print(used_probs, [tf.shape(old_actions)], message = "old_actions has shape ")
        #used_probs = tf.Print(used_probs, [tf.shape(multiplier)], message = "multiplier has shape ")
        #used_probs = tf.Print(used_probs, [tf.shape(used_probs)], message = "used_probs has shape ")
        #used_probs = tf.Print(used_probs, [tf.shape(old_values)], message = "old_values has shape ")
        
        
        loss = -tf.reduce_mean(used_probs/tf.stop_gradient(used_probs) * old_advantages)
        #loss = tf.Print(loss, [loss], message = "loss = ")
        return  old_actions, old_advantages, loss
                                         
    def train(self, old_states, old_actions, old_advantages):
        self.sess.run(self.optimize, 
                      feed_dict = {self.state: old_states,
                                  self.old_actions: old_actions,
                                  self.old_advantages: old_advantages})
    
    def predict(self, state):
        return self.sess.run(self.probs,
                             feed_dict = {self.state: state})
    
    def create_value_network(self): 
        # feedforward / recurrent model to value
        l1 = tflearn.fully_connected(self.state, 200, activation='relu')
        l2 = tflearn.fully_connected(l1, 150)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        value = tflearn.fully_connected(l2, 1, weights_init=w_init)
        return value
    
    def create_value_loss(self):
         # Obtained from the target networks
        old_hindsight_values = tf.placeholder(tf.float32, [None, 1])
        # Define loss and optimization Op
        loss = tflearn.mean_square(old_hindsight_values, self.value)
        return old_hindsight_values, loss
        
    def train_value(self, old_states, old_hindsight_values):
        self.sess.run(self.optimize_value, 
                      feed_dict = {self.state: old_states,
                                   self.old_hindsight_values: old_hindsight_values})
    def predict_value(self, state):
        return self.sess.run(self.value,
                             feed_dict = {self.state: state})