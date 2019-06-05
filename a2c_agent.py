
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

from utils import create_layers

conv2d, dense, flatten = create_layers()


class A2CAgent():

    def __init__(self, name, trainable, sess, input_shape, action_size, lr, GAMMA, LAMBDA, max_grad_norm, ent_coef, vf_coef, loadModel):
        self.sess = sess
        self.input_shape = input_shape
        self.action_size = action_size
        self.lr = lr
        self.GAMMA = GAMMA
        self.LAMBDA = LAMBDA
        self.max_grad_norm = max_grad_norm
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.trainable = trainable
        self.name = name

        with tf.variable_scope(self.name+"/", reuse=tf.AUTO_REUSE):
            self.state_input = tf.placeholder(tf.float64, shape=(
                None, *self.input_shape), name="state")
            self.actions_input = tf.placeholder(tf.uint8, shape=(
                None,), name="actions_input")
            self.cumulative_rewards_input = tf.placeholder(
                tf.float64, shape=(None,), name="cumulative_rewards_input")
            self.state_values_input = tf.placeholder(
                tf.float64, shape=(None,), name="state_values_input")

        self.actor_out, self.critic_out, self.saver = self._build_agent(
            self.trainable)

        if self.trainable:
            self.train_model, self.actor_loss, self.critic_loss, self.entropy = self._build_ops()

        if loadModel:
            self.load_weights()

    def _build_agent(self, trainable=True):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            self.sess.run(tf.local_variables_initializer())

            shared_net = conv2d(self.state_input, 3, 32,
                                "VALID", strides=2, activation=tf.nn.relu, trainable=trainable)
            shared_net = conv2d(shared_net, 3, 32, "VALID",
                                strides=2, activation=tf.nn.relu, trainable=trainable)
            shared_net = conv2d(shared_net, 3, 32, "VALID",
                                strides=1, activation=tf.nn.relu, trainable=trainable)
            shared_net = conv2d(shared_net, 3, 32, "VALID",
                                strides=1, activation=tf.nn.relu, trainable=trainable)

            shared_net = flatten(shared_net)

            actor_out = dense(
                shared_net, 512, activation=tf.nn.relu, trainable=trainable)
            actor_out = dense(actor_out, self.action_size,
                              activation=tf.nn.softmax, trainable=trainable)

            critic_out = dense(
                shared_net, 512, activation=tf.nn.relu, trainable=trainable)
            critic_out = dense(critic_out, 1, trainable=trainable)

            saver = tf.train.Saver(var_list=tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))

            return actor_out, critic_out, saver

    def _build_ops(self):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            advantages = self.cumulative_rewards_input - self.critic_out

            critic_loss = tf.reduce_mean(tf.square(advantages))

            action_one_hot = tf.one_hot(self.actions_input, self.action_size, dtype=tf.float64)
            logs = tf.log(tf.reduce_sum(
                self.actor_out * action_one_hot, axis=1))
            actor_loss = tf.reduce_mean(logs * advantages)

            entropy = tf.reduce_mean(self.actor_out * tf.log(self.actor_out))

            loss = self.vf_coef * critic_loss - actor_loss - self.ent_coef * entropy

            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

            params = tf.trainable_variables()
            grads = tf.gradients(loss, params)

            if self.max_grad_norm is not None:
                # Clip the gradients (normalize)
                grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
                grads = list(zip(grads, params))

            train_model = optimizer.apply_gradients(grads)

            return train_model, actor_loss, critic_loss, entropy

    def sync_from_model(self, from_model):
        update_weights = [tf.assign(new, old) for (new, old) in zip(
            tf.trainable_variables(self.name), tf.trainable_variables(from_model.name))]
        self.sess.run(update_weights)

    def save_weights(self):
        self.saver.save(self.sess, f"./{self.name}.ckpt")

    def load_weights(self):
        self.saver.restore(self.sess, f"./{self.name}.ckpt")

    def get_action_and_value(self, state):
        return self.sess.run([self.actor_out, self.critic_out], feed_dict={self.state_input: state})

    def train(self, states, actions, cumulative_rewards):
        _, critic_loss, actor_loss, entropy = self.sess.run([self.train_model, self.critic_loss, self.actor_loss, self.entropy], {
                                                   self.state_input: states, self.actions_input: actions, self.cumulative_rewards_input: cumulative_rewards})
        print(f"critic_loss:{critic_loss}, actor_loss:{actor_loss}, entropy:{entropy}")
