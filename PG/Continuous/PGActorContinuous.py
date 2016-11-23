import tensorflow as tf
import tflearn
import numpy as np

class PGActorContinuous(object):

    def __init__(self, sess, state_dim, action_dim, action_bound, actor_learning_rate, critic_learning_rate):  
        
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        
        # Create the action graph
        self.std_param, self.state, self.means, self.stds = self.create_actor_network()
        self.actor_params = tf.trainable_variables()
        print "There are", len(self.actor_params), "actor params"
        
        # Create the loss calculation and optimizer graph
        self.old_actions, self.old_advantages, self.loss = self.create_loss()
        self.actor_optimizer = tf.train.AdamOptimizer(self.actor_learning_rate)
        self.optimize = self.actor_optimizer.minimize(self.loss, var_list = self.actor_params)
        
        # Create value graph
        self.value = self.create_value_network()
        self.value_params = tf.trainable_variables()[len(self.actor_params):]
        print "There are", len(self.value_params), "value params"
        # Create value loss calculation and optimizer graph
        self.old_returns, self.value_loss = self.create_value_loss()
        self.value_optimizer = tf.train.AdamOptimizer(self.critic_learning_rate)
        self.optimize_value = self.value_optimizer.minimize(self.value_loss, var_list = self.value_params)
        
        
    def create_actor_network(self):
        # stddevs (not functions of state)
        std_param = tf.Variable([-0.5])
        stds = tf.exp(std_param)
        
        # state placeholders (these could be sequences for recurrent model)
        state = tflearn.input_data(shape=[None, self.s_dim])
        # feedforward / recurrent model to distribution parameters
        l1 = tflearn.fully_connected(state, 40, activation='relu')
        #l2 = tflearn.fully_connected(l1, 30, activation='relu')
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        unscaled_means = tflearn.fully_connected(l1, self.a_dim, activation='tanh', weights_init=w_init)
        means = tf.mul(unscaled_means, self.action_bound)
        
        return std_param, state, means, stds
    
    def create_loss(self):
        old_actions = tflearn.input_data(shape=[None, self.a_dim])
        old_advantages = tflearn.input_data(shape=[None, 1])
        batch_size = tf.to_int32(
            tf.reduce_sum(tf.reduce_sum(self.means * 0, 1) + 1))
        stds_tiled = tf.to_float(tf.tile(self.stds, [batch_size]))
       
        
        # Create the raw gaussian distribution for an action
        dists = tf.contrib.distributions.Normal(tf.squeeze(self.means, [1]), tf.to_float(stds_tiled))
        unnormed_probs = (dists.pdf(tf.squeeze(old_actions, [1])))
        # Modify the probs to account for truncation
        total_mass = dists.cdf(tf.to_float(self.action_bound))-dists.cdf(tf.to_float(-self.action_bound))
        probs = tf.expand_dims(unnormed_probs/total_mass, 1)
        losses = -probs/tf.stop_gradient(probs) * old_advantages
        loss = tf.reduce_mean(losses)
        
        loss = tf.Print(loss, [tf.shape(self.means)], message = "self.means has shape: ")
        loss = tf.Print(loss, [tf.shape(stds_tiled)], message = "stds_tiled has shape: ")
        loss = tf.Print(loss, [tf.shape(old_advantages)], message = "old_advantages has shape: ")
        loss = tf.Print(loss, [tf.shape(old_actions)], message = "old_actions has shape: ")
        loss = tf.Print(loss, [tf.shape(unnormed_probs)], message = "unnormed_probs has shape: ")
        loss = tf.Print(loss, [tf.shape(total_mass)], message = "total_mass has shape: ")
        loss = tf.Print(loss, [tf.shape(probs)], message = "probs has shape: ")
        loss = tf.Print(loss, [tf.shape(losses)], message = "losses has shape: ")
        loss = tf.Print(loss, [tf.shape(loss)], message = "loss has shape: ")
        
        loss = tf.Print(loss, [old_actions], message = "old_actions = ")
        loss = tf.Print(loss, [unnormed_probs], message = "unnormed_probs = ")
        loss = tf.Print(loss, [total_mass], message = "total mass = ")
        loss = tf.Print(loss, [probs], message = "probs = ")
        loss = tf.Print(loss, [old_advantages], message = "old_advantages = ")
        loss = tf.Print(loss, [self.means], message = "self.means = ")
        loss = tf.Print(loss, [self.stds], message = "self.stds = ")
        loss = tf.Print(loss, [stds_tiled], message = "stds_tiled = ")
        #loss = tf.Print(loss, [probs], message = "probs = ")
        loss = tf.Print(loss, [losses], message = "losses = ")
        
        loss = tf.Print(loss, [loss], message = "loss = ")
        
        return old_actions, old_advantages,  loss
                                         
    def train(self, old_states, old_actions, old_advantages):
        self.sess.run(self.optimize, 
                      feed_dict = {self.state: old_states,
                                  self.old_actions: old_actions,
                                  self.old_advantages: old_advantages})
    
    def predict(self, state):
        return self.sess.run([self.means, self.stds],
                             feed_dict = {self.state: state})
    
    def create_value_network(self): 
        # feedforward / recurrent model to value
        l1 = tflearn.fully_connected(self.state, 40, activation='relu')
        l2 = tflearn.fully_connected(l1, 30, activation = 'relu')
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        value = tflearn.fully_connected(l2, 1, activation = 'linear', weights_init=w_init)
        return value
    
    def create_value_loss(self):
         # Obtained from the target networks
        old_returns = tf.placeholder(tf.float32, [None, 1])
        # Define loss and optimization Op
        
        loss = tflearn.mean_square(old_returns, self.value)
        loss = tf.Print(loss, [tf.shape(old_returns)], message = "old_returns has shape: ")
        loss = tf.Print(loss, [tf.shape(self.value)], message = "value has shape: ")
        loss = tf.Print(loss, [old_returns], message = "old_returns = ")
        loss = tf.Print(loss, [self.value], message = "value = ")
        loss = tf.Print(loss, [loss], message = "loss = ")
        
        return old_returns, loss
    
    def train_value(self, old_states, old_returns):
        self.sess.run(self.optimize_value, 
                      feed_dict = {self.state: old_states,
                                   self.old_returns: old_returns})
    def predict_value(self, state):
        return self.sess.run(self.value,
                             feed_dict = {self.state: state})
    
   