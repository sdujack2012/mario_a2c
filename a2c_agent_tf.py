
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import random


def create_layers():
    layer_count = 0

    def conv2d(input_layer, kernel_size, filters, padding, strides=1, activation=None, trainable=True):
        nonlocal layer_count
        input_shape = input_layer.get_shape()
        input_dim = len(input_shape)

        W = tf.get_variable(f'w_{layer_count}', shape=(
            kernel_size, kernel_size, input_shape[input_dim - 1], filters), initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
        B = tf.get_variable(f'b_{layer_count}', shape=(
            filters), initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
        layer_count += 1
        # Conv2D wrapper, with bias and relu activation
        out = tf.nn.conv2d(input_layer, W, strides=[
                           1, strides, strides, 1], padding=padding)
        out = tf.nn.bias_add(out, B)
        return out if activation == None else activation(out)

    def dense(input_layer, output, activation=None, trainable=True):
        nonlocal layer_count
        W = tf.get_variable(f'w_{layer_count}', shape=(input_layer.get_shape()[
                            1], output), initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
        B = tf.get_variable(f'b_{layer_count}', shape=(
            output), initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
        layer_count += 1

        out = tf.add(tf.matmul(input_layer, W), B)
        return out if activation == None else activation(out)

    def flatten(input_layer):
        shape = int(np.prod(input_layer.get_shape()[1:]))
        out = tf.reshape(input_layer, [-1, shape])
        return out

    return conv2d, dense, flatten


conv2d, dense, flatten = create_layers()


class A2CAgent():

    def __init__(self, sess, input_shape, action_size, previous_actions_size, lr, GAMMA, LAMBDA, loadModel):
        self.sess = sess
        self.input_shape = input_shape
        self.action_size = action_size
        self.previous_actions_size = previous_actions_size
        self.lr = lr
        self.GAMMA = GAMMA
        self.LAMBDA = LAMBDA

        self.image_input = tf.placeholder(tf.float32, shape=(
            None, *self.input_shape), name="image_input")
        self.previous_actions_input = tf.placeholder(tf.float32, shape=(
            None, self.previous_actions_size), name="previous_actions_input")
        self.actions_input = tf.placeholder(tf.float32, shape=(
            None, self.action_size), name="actions_input")
        self.advantages_input = tf.placeholder(
            tf.float32, shape=(None,), name="advantages_input")
        self.discounted_rewards_input = tf.placeholder(
            tf.float32, shape=(None,), name="discounted_rewards_input")

        #self.old_model_output_pl = K.placeholder(shape=(None, self.action_size))

        self.actor_out, self.critic_out, self.saver = self._build_agent(
            "model")
        self.old_actor_out, _, _ = self._build_agent("old_model")

        self.train_model, self.actor_loss, self.critic_loss = self._build_ops()

        # Initializing the variables
        self.sess.run(tf.global_variables_initializer())

        if loadModel:
            self.load_weights()

        self.sync_old_model()

    def _build_agent(self, name, trainable=True):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            self.sess.run(tf.local_variables_initializer())

            shared_net = conv2d(self.image_input, 3, 32,
                                "VALID", strides=2, activation=tf.nn.relu, trainable=trainable)
            shared_net = conv2d(shared_net, 3, 32, "VALID",
                                strides=2, activation=tf.nn.relu, trainable=trainable)
            shared_net = conv2d(shared_net, 3, 32, "VALID",
                                strides=1, activation=tf.nn.relu, trainable=trainable)
            shared_net = conv2d(shared_net, 3, 32, "VALID",
                                strides=1, activation=tf.nn.relu, trainable=trainable)

            shared_net = flatten(shared_net)
            shared_net = tf.concat(
                [shared_net, self.previous_actions_input], 1)

            actor_out = dense(shared_net, 512, activation=tf.nn.relu, trainable=trainable)
            actor_out = dense(actor_out, self.action_size,
                              activation=tf.nn.softmax, trainable=trainable)

            critic_out = dense(shared_net, 512, activation=tf.nn.relu, trainable=trainable)
            critic_out = dense(critic_out, 1, trainable=trainable)

            saver = tf.train.Saver(var_list=tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=name))

            return actor_out, critic_out, saver

    def _build_ops(self):
        with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):

            critic_loss = tf.losses.mean_squared_error(
                tf.squeeze(self.critic_out), self.discounted_rewards_input)

            neglogs = 0 - \
                tf.log(tf.reduce_sum(self.actor_out * self.actions_input, axis=1))
            actor_loss = tf.reduce_mean(neglogs * self.advantages_input)

            entropy = tf.reduce_mean(self.actor_out * tf.log(self.actor_out))

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            opt = tf.train.AdamOptimizer(learning_rate=self.lr)

            train_model = opt.minimize(
                loss, var_list=tf.trainable_variables("model"))

            return train_model, actor_loss, critic_loss

    def sync_old_model(self):
        update_weights = [tf.assign(new, old) for (new, old) in zip(
            tf.trainable_variables('old_model'), tf.trainable_variables('model'))]
        self.sess.run(update_weights)

    def save_weights(self):
        self.saver.save(self.sess, "./model.ckpt")

    def load_weights(self):
        self.saver.restore(self.sess, "./model.ckpt")

    def get_action_and_value(self, image_input, previous_actions_input):
        return self.sess.run([self.actor_out, self.critic_out], feed_dict={self.image_input: image_input, self.previous_actions_input: previous_actions_input})

    def train(self, states, previous_actions, state_values, next_state_values, actions, rewards, dones):
        """ Update actor and critic networks from experience
        """
        # Compute discounted rewards and Advantage (TD. Error)
        discounted_rewards = np.array(self.get_discount_rewards(rewards))
        advantages = np.array(self.get_gaes(
            rewards, state_values, next_state_values, self.GAMMA, self.LAMBDA))
        # Networks optimization
        return self.sess.run([self.train_model, self.critic_loss, self.actor_loss], {self.image_input: states, self.actions_input: actions, self.previous_actions_input: previous_actions, self.advantages_input: advantages, self.discounted_rewards_input: discounted_rewards})

    def process_episode_experiences(self, experiences):
        state_index = 0
        previous_action_index = 1
        action_index = 2
        state_value_index = 3
        reward_index = 4
        done_index = 5

        states = [experience[state_index] for experience in experiences]
        previous_actions = [experience[previous_action_index]
                            for experience in experiences]
        state_values = [experience[state_value_index]
                        for experience in experiences]
        next_state_values = state_values[1:] + [0]

        actions = [experience[action_index] for experience in experiences]
        rewards = [experience[reward_index] for experience in experiences]
        dones = [experience[done_index] for experience in experiences]

        return states, previous_actions, state_values, next_state_values, actions, rewards, dones

    def train_with_experiences(self, episode_experiences):
        all_states = []
        all_previous_actions = []
        all_state_values = []
        all_next_state_values = []
        all_actions = []
        all_rewards = []
        all_dones = []

        for experiences in episode_experiences:
            states, previous_actions, state_values, next_state_values, actions, rewards, dones = self.process_episode_experiences(
                experiences)
            all_states += states
            all_previous_actions += previous_actions
            all_state_values += state_values
            all_next_state_values += next_state_values
            all_actions += actions
            all_rewards += rewards
            all_dones += dones

        length = len(all_states)
        shuffled = list(range(length))
        random.shuffle(shuffled)
        all_states = np.array(all_states)[shuffled]
        all_previous_actions = np.array(all_previous_actions)[shuffled]
        all_state_values = np.array(all_state_values)[shuffled]
        all_next_state_values = np.array(all_next_state_values)[shuffled]
        all_actions = np.array(all_actions)[shuffled]
        all_rewards = np.array(all_rewards)[shuffled]
        all_dones = np.array(all_dones)[shuffled]

        bacth_size = 32
        bacth_index = range(0, len(all_states))
        bacthes = np.array_split(bacth_index, bacth_size)
        for batch in bacthes:
            _, critic_loss, actor_loss = self.train(all_states[batch], all_previous_actions[batch], all_state_values[batch],
                                                    all_next_state_values[batch], all_actions[batch], all_rewards[batch], all_dones[batch])
            print(f"critic_loss:{critic_loss}, actor_loss:{actor_loss}")

    def get_discount_rewards(self, r):
        discounted_r = np.zeros(len(r))
        cumul_r = 0
        for t in reversed(range(len(r))):
            cumul_r = r[t] + cumul_r * self.GAMMA
            discounted_r[t] = cumul_r
        return discounted_r

    # We are defining the function to get the Generalized Advantage Estimation
    def get_gaes(self, rewards, state_values, next_state_values, GAMMA, LAMBDA):
        gaes = np.array([r_t + GAMMA * next_v - v for r_t, next_v, v in zip(rewards, next_state_values, state_values)])
        for t in reversed(range(len(gaes) - 1)):
            gaes[t] = gaes[t] + LAMBDA * GAMMA * gaes[t + 1]
        return gaes