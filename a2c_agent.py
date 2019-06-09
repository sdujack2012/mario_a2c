
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

class A2CAgent():

    def __init__(self, name, isMaster, sess, input_shape, action_size, lr, GAMMA, LAMBDA, max_grad_norm, ent_coef, vf_coef, loadModel):
        self.sess = sess
        self.input_shape = input_shape
        self.action_size = action_size
        self.lr = lr
        self.GAMMA = GAMMA
        self.LAMBDA = LAMBDA
        self.max_grad_norm = max_grad_norm
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.name = name
        self.isMaster = isMaster
        self.state = tf.placeholder(tf.uint8, shape=(None, *self.input_shape), name="state")
        self.actions = tf.placeholder(tf.uint8, shape=(None,), name="actions")
        self.cumulative_rewards = tf.placeholder(tf.float32, shape=(None,), name="cumulative_rewards")
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name="advantages")
        self.episode_rewards = tf.placeholder(tf.float32, shape=(), name="episode_rewards")
        self.max_episode_rewards = tf.placeholder(tf.float32, shape=(), name="max_episode_rewards")
        self.old_policy = tf.placeholder(tf.float32, shape=(None, self.action_size), name="old_policy")
        self.policy, self.value, self.saver = self._build_agent()

        if isMaster:
            self.train_model, self.policy_loss, self.value_loss, self.entropy = self._build_ops()
        
        self.sess.run(tf.initialize_variables(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)))

        if loadModel:
            self.load_weights()
            
    def _build_agent(self):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            
            x_conv = tf.cast(self.state, tf.float32) / 255.0
            conv1 = tf.layers.conv2d(inputs=x_conv, filters=32, kernel_size=[8,8], strides=(4, 4), activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[4,4], strides=(2, 2), activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3,3], strides=(1, 1), activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            
            conv_out = tf.contrib.layers.flatten(conv3)
            shared_dense = tf.layers.dense(conv_out, 512, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        
            policy_logits = tf.layers.dense(shared_dense, self.action_size, kernel_initializer=tf.contrib.layers.xavier_initializer())
            policy = tf.nn.softmax(policy_logits)
            
            value = tf.layers.dense(shared_dense, 1, kernel_initializer=tf.contrib.layers.xavier_initializer())

            saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

            return policy, value, saver

    def _build_ops(self):
        with tf.variable_scope(self.name):
            local_vars = tf.trainable_variables()

        with tf.name_scope("{}_gradient".format(self.name)):
            value_loss = 0.5 * tf.reduce_mean(tf.square(tf.squeeze(self.value) - self.cumulative_rewards))
            tf.summary.scalar('value_loss', value_loss)

            action_one_hot = tf.one_hot(self.actions, self.action_size, dtype=tf.float32)
            
            neg_log_policy = -tf.log(tf.clip_by_value(self.policy, 1e-7, 1))
            
            log_policy = tf.log(tf.reduce_sum(self.policy * action_one_hot, axis=1))
            old_log_policy = tf.log(tf.reduce_sum(self.old_policy * action_one_hot, axis=1))

            ratio = tf.exp(log_policy - old_log_policy) # pnew / pold
            surr1 = ratio * self.advantages # surrogate from conservative policy iteration
            surr2 = tf.clip_by_value(ratio, 1.0 - 0.2, 1.0 + 0.2) * self.advantages #

            policy_loss = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)

            
        
            tf.summary.scalar('policy_loss', policy_loss)

            entropy = tf.reduce_mean(tf.reduce_sum(self.policy * neg_log_policy, axis=1))
        
            tf.summary.scalar('entropy_loss', entropy)

            total_loss = self.vf_coef * value_loss + policy_loss - self.ent_coef * entropy

            tf.summary.scalar('total_loss', policy_loss)

            tf.summary.scalar('episode_rewards', self.episode_rewards)
            tf.summary.scalar('max_episode_rewards', self.max_episode_rewards)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

            grads = tf.gradients(total_loss, local_vars)

            if self.max_grad_norm is not None:
                grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
                grads = list(zip(grads, local_vars))

            l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
            for gradient, variable in grads:
                tf.summary.histogram("gradients/" + variable.name, l2_norm(gradient))
                tf.summary.histogram("variables/" + variable.name, l2_norm(variable))

            train_model = optimizer.apply_gradients(grads)

            return train_model, policy_loss, value_loss, entropy

    def create_sync_ops(self, from_model):
        return [tf.assign(to_var, from_var) for (to_var, from_var) in zip(
            tf.trainable_variables(self.name), tf.trainable_variables(from_model.name))]

    def save_weights(self):
        self.saver.save(self.sess, f"./{self.name}.ckpt")

    def load_weights(self):
        self.saver.restore(self.sess, f"./{self.name}.ckpt")

    def get_actions_and_values(self, state):
        return self.sess.run([self.policy, self.value], feed_dict={self.state: state})

    def get_action(self, state):
        return self.sess.run(self.policy, feed_dict={self.state: state})

    def get_value(self, state):
        return self.sess.run(self.value, feed_dict={self.state: state})

    def train(self, states, actions, cumulative_rewards, advantages, old_policy):
        _, value_loss, policy_loss, entropy = self.sess.run([self.train_model, self.value_loss, self.policy_loss, self.entropy], {
                                                   self.state: states, self.actions: actions, self.cumulative_rewards: cumulative_rewards, self.advantages: advantages, self.old_policy: old_policy})
        print(f"critic_loss:{value_loss}, actor_loss:{policy_loss}, entropy:{entropy}")

    def get_summary(self, states, actions, cumulative_rewards, advantages, max_episode_rewards, episode_rewards, old_policy):
        summaries_op = tf.summary.merge_all()
        summary = self.sess.run(summaries_op, feed_dict={
                                                   self.max_episode_rewards: max_episode_rewards, self.episode_rewards: episode_rewards, self.state: states, self.actions: actions, self.cumulative_rewards: cumulative_rewards, self.advantages: advantages, self.old_policy: old_policy})
        return summary
            