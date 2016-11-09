import tensorflow as tf
import tflearn

class ActorPair(object):
    
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        
        # Create the main network
        self.state, self.unscaled_action, self.action = self.create_actor_network()
        # Create a reference to the params
        self.network_params = tf.trainable_variables()
        # Create the target network
        self.target_state, self.target_unscaled_action, self.target_action = self.create_actor_network()
        # Reference the target params
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]
        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)
        
        # dQ/da is provided by critic
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
        # dQ/dtheta = dQ/da da/dtheta
        self.actor_gradients = tf.gradients(self.action, self.network_params, -self.action_gradient)
        # Create the optimization op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))
        
        # Create an interpolation op for the target network
        self.update_target_network_params = \
        [self.target_network_params[i].assign(tf.mul(self.network_params[i], self.tau) + \
            tf.mul(self.target_network_params[i], 1. - self.tau))
            for i in range(len(self.target_network_params))]
    
    def get_action(state):
        # call to to sess.run on self.action_var
        return action
    
    def create_actor_network(self):
        # state placeholders (these could be sequences for recurrent model)
        state = tflearn.input_data(shape=[None, self.s_dim])
        
        # feedforward / recurrent model to action 
        
        l1 = tflearn.fully_connected(state, 400, activation='relu')
        l2 = tflearn.fully_connected(l1, 300, activation='relu')
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        unscaled_action = tflearn.fully_connected(l2, self.a_dim, activation='tanh', weights_init=w_init)
        action = tf.mul(unscaled_action, self.action_bound) # Scale output to -action_bound to action_bound
        return state, unscaled_action, action 
    
    
    def train(self, state, a_gradient):
        self.sess.run(self.optimize, 
                      feed_dict = {self.state: state,
                                   self.action_gradient : a_gradient
                                  }
                     )
    
    def predict(self, state):
        return self.sess.run(self.action,
                             feed_dict = {self.state: state}
                            )
    
    def predict_target(self, state):
        return self.sess.run(self.target_action,
                             feed_dict = {self.target_state: state}
                            )
    
    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
        
    def get_num_trainable_vars(self):
        return self.num_trainable_vars