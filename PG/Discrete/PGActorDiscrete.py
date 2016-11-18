import tensorflow as tf
import tflearn
import numpy as np

class PGActorDiscrete(object):

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate):  
        
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        
        # Create the action graph
        self.state, self.probs = self.create_actor_network()
        
        # Create the loss calculation and optimizer graph
        self.old_actions, self.old_values, self.loss = self.create_loss()
        self.network_params = tf.trainable_variables()
        self.loss_gradients = tf.gradients(self.loss, self.network_params)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.loss_gradients, self.network_params))

    def create_actor_network(self):
        # state placeholders (these could be sequences for recurrent model)
        state = tflearn.input_data(shape=[None, self.s_dim])
        # feedforward / recurrent model to action 
        
        #l1 = tflearn.fully_connected(state, 30, activation='relu')
        l2 = tflearn.fully_connected(state, 10, activation='relu')
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
        old_values = tflearn.input_data(shape=[None, 1])
        
        #vars = self.stds**2
        #probs = tf.exp(-((old_actions-self.means)**2)/(2*vars))/tf.sqrt(2*vars * np.pi)
        multiplier = tf.concat(1, [(1-old_actions), old_actions])
        used_probs = tf.reduce_mean(self.probs * multiplier, reduction_indices = 1, keep_dims = True)
        #used_probs = tf.Print(used_probs, [tf.shape(old_actions)], message = "old_actions has shape ")
        #used_probs = tf.Print(used_probs, [tf.shape(multiplier)], message = "multiplier has shape ")
        #used_probs = tf.Print(used_probs, [tf.shape(used_probs)], message = "used_probs has shape ")
        #used_probs = tf.Print(used_probs, [tf.shape(old_values)], message = "old_values has shape ")
        
        
        loss = -tf.reduce_sum(tf.log(used_probs) * old_values)
        #loss = tf.Print(loss, [loss], message = "loss = ")
        return  old_actions, old_values, loss
                                         
    def train(self, old_states, old_actions, old_values):
        self.sess.run(self.optimize, 
                      feed_dict = {self.state: old_states,
                                  self.old_actions: old_actions,
                                  self.old_values: old_values})
    
    def predict(self, state):
        return self.sess.run(self.probs,
                             feed_dict = {self.state: state})