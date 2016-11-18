import tensorflow as tf
import tflearn

class CriticPair(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        
        
        # Create the main network
        self.state, self.action, self.value = self.create_critic_network()
        # Create a reference to the params
        self.network_params = tf.trainable_variables()[num_actor_vars:]
        # Create the target network
        self.target_state, self.target_action, self.target_value = self.create_critic_network()
        # Reference the target params
        self.target_network_params = tf.trainable_variables()[num_actor_vars+len(self.network_params):]
        
        # Obtained from the target networks
        self.hindsight_q_value = tf.placeholder(tf.float32, [None, 1])
        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.hindsight_q_value, self.value)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        # Get the gradient of the net w.r.t. the action
        self.action_grads = tf.gradients(self.value, self.action)
        
        # Create an interpolation op for the target
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.mul(self.network_params[i], self.tau) + \
                tf.mul(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]
       
    def create_critic_network(self): # TODO 
        # state placeholders (these could be sequences for recurrent model)
        state = tflearn.input_data(shape=[None, self.s_dim])
        # action placeholder
        action = tflearn.input_data(shape=[None, self.a_dim])
        
        # feedforward / recurrent model to value
        
        l1 = tflearn.fully_connected(state, 400, activation='relu')
        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(l1, 300)
        t2 = tflearn.fully_connected(action, 300)
        l2 = tflearn.activation(tf.matmul(l1,t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')
        # linear layer connected to 1 output representing Q(s,a) 
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        value = tflearn.fully_connected(l2, 1, weights_init=w_init)
        return state, action, value
    
    
    def train(self, state, action, hindsight_q_value):
        return self.sess.run([self.value, self.optimize], 
                      feed_dict = {self.state: state,
                                   self.action: action,
                                   self.hindsight_q_value: hindsight_q_value})
        
    
    def predict(self, state, action):
        return self.sess.run(self.value,
                             feed_dict = {self.state: state,
                                          self.action: action
                                         })
    
    def predict_target(self, state, action):
        return self.sess.run(self.target_value,
                             feed_dict = {self.target_state: state,
                                          self.target_action: action})
    def action_gradients(self, state, actions):
        return self.sess.run(self.action_grads, 
                             feed_dict = {self.state: state,
                                          self.action: actions})
    
    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
        