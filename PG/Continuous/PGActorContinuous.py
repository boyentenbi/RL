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
        std_params = tf.Variable(np.zeros([self.a_dim]))
        stds = tf.to_float(tf.exp(std_params))*self.action_bound
        
        # state placeholders (these could be sequences for recurrent model)
        state = tflearn.input_data(shape=[None, self.s_dim])
        # feedforward / recurrent model to distribution parameters
        state_norm = tflearn.layers.normalization.batch_normalization(state)
        l1 = tflearn.fully_connected(state_norm, 30, activation='relu')
        l1_norm = tflearn.layers.normalization.batch_normalization(l1) 
        l2 = tflearn.fully_connected(l1_norm, 30, activation='relu') + l1_norm
        l2_norm = tflearn.layers.normalization.batch_normalization(l2)
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        unscaled_means = tflearn.fully_connected(l2_norm, self.a_dim, activation='tanh', weights_init=w_init)
        means = tf.mul(unscaled_means, self.action_bound)
        
        return std_params, state, means, stds
    
    def create_loss(self):
        old_actions = tflearn.input_data(shape=[None, self.a_dim])
        old_advantages = tflearn.input_data(shape=[None, 1])
        #batch_size = tf.to_int32(
            #tf.reduce_sum(tf.reduce_sum(self.means * 0, 1) + 1))
        #stds_tiled = tf.to_float(tf.tile(self.stds, [batch_size]))
       
        
        # Create the raw gaussian distribution for an action
        means = tf.Print(self.means, [tf.shape(self.means)], "self.means has shape: ")
        
        stds = tf.Print(self.stds, [tf.shape(self.stds)], "self.stds has shape: ")
        stds = tf.Print(stds, [tf.shape(stds + means)], "stds + means has shape: " )
        dists = tf.contrib.distributions.Normal(means, stds)
        unnormed_probs = tf.reduce_prod(dists.pdf(old_actions), 1, keep_dims=True)
        unnormed_probs = tf.Print(unnormed_probs, [tf.shape(unnormed_probs)], "unnormed_probs has shape")
        # Modify the probs to account for truncation
        masses = dists.cdf(tf.to_float(self.action_bound))-dists.cdf(tf.to_float(-self.action_bound))
        masses = tf.Print(masses, [tf.shape(masses)], message = "masses has shape: ")
        masses = tf.Print(masses, [masses], message = "masses = ")
        total_mass = tf.reduce_prod(masses, 1, keep_dims=True)
        total_mass = tf.Print(total_mass, [tf.shape(total_mass)], message = "total_mass has shape: ")
        probs = tf.truediv(unnormed_probs,total_mass)
        probs = tf.Print(probs, [tf.shape(probs)], message = "probs has shape: ")
        losses = old_advantages * -probs/tf.stop_gradient(probs)
        losses = tf.Print(losses, [tf.shape(losses)], message = "losses has shape: ")
        loss = tf.reduce_mean(losses)
        loss = tf.Print(loss, [tf.shape(loss)], message = "loss has shape: ")
        
        loss = tf.Print(loss, [tf.shape(old_advantages)], message = "old_advantages has shape: ")
        loss = tf.Print(loss, [tf.shape(old_actions)], message = "old_actions has shape: ")
        
        loss = tf.Print(loss, [old_actions], message = "old_actions = ")
        loss = tf.Print(loss, [unnormed_probs], message = "unnormed_probs = ")
        loss = tf.Print(loss, [total_mass], message = "total mass = ")
        loss = tf.Print(loss, [probs], message = "probs = ")
        loss = tf.Print(loss, [old_advantages], message = "old_advantages = ")
        loss = tf.Print(loss, [self.means], message = "self.means = ")
        loss = tf.Print(loss, [self.stds], message = "self.stds = ")
        #loss = tf.Print(loss, [stds_tiled], message = "stds_tiled = ")
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
        state_norm = tflearn.layers.normalization.batch_normalization(self.state)
        l1 = tflearn.fully_connected(state_norm, 30, activation='relu')
        l1_norm = tflearn.layers.normalization.batch_normalization(l1)
        l2 = tflearn.fully_connected(l1_norm, 30, activation = 'relu') + l1_norm
        l2_norm = tflearn.layers.normalization.batch_normalization(l2)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        value = tflearn.fully_connected(l2_norm, 1, activation = 'linear', weights_init=w_init)
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
    
   