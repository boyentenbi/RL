import tflearn
import tensorflow as tf
def class PGCriticDiscrete(object):
    def __init__(self):
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