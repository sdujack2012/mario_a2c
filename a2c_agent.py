
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random


class A2CAgent():

    def __init__(self, name, is_train_model, sess, input_shape, action_size, lr, GAMMA, LAMBDA, max_grad_norm, ent_coef, vf_coef, clip_range, loadModel):
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
        self.is_train_model = is_train_model
        self.clip_range = clip_range

        self.state = tf.placeholder(tf.uint8, shape=(
            None, *self.input_shape), name="state")
        self.actions = tf.placeholder(tf.uint8, shape=(None,), name="actions")
        self.rewards = tf.placeholder(
            tf.float32, shape=(None,), name="rewards")
        self.advantages = tf.placeholder(
            tf.float32, shape=(None,), name="advantages")

        self.old_policy = tf.placeholder(tf.float32, shape=(
            None, self.action_size), name="old_policy")
        self.old_values = tf.placeholder(
            tf.float32, shape=(None,), name="old_values")

        self.episode_rewards = tf.placeholder(
            tf.float32, shape=(), name="episode_rewards")
        self.max_episode_rewards = tf.placeholder(
            tf.float32, shape=(), name="max_episode_rewards")

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.x_conv = tf.cast(self.state, tf.float32) / 255.0
            self.conv1 = tf.layers.conv2d(inputs=self.x_conv, filters=32, kernel_size=[8, 8], strides=(
                4, 4), activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, kernel_size=[4, 4], strides=(
                2, 2), activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, kernel_size=[3, 3], strides=(
                1, 1), activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())

            self.conv_out = tf.contrib.layers.flatten(self.conv3)
            self.shared_dense = tf.layers.dense(
                self.conv_out, 512, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

            self.policy_logits = tf.layers.dense(
                self.shared_dense, self.action_size, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.policy = tf.nn.softmax(self.policy_logits)

            self.values = tf.layers.dense(
                self.shared_dense, 1, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.local_vars = tf.trainable_variables()
            self.saver = tf.train.Saver(
                var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        if is_train_model:
            with tf.name_scope("{}_gradient".format(self.name)):
                cliped_values = self.old_values + \
                    tf.clip_by_value(
                        self.values - self.old_values, -self.clip_range, self.clip_range)
                
                value_loss1 = tf.square(tf.squeeze(self.values) - self.rewards)
                value_loss2 = tf.square(
                    tf.squeeze(cliped_values) - self.rewards)

                self.value_loss = 0.5 * \
                    tf.reduce_mean(tf.maximum(value_loss1, value_loss2))

                action_one_hot = tf.one_hot(
                    self.actions, self.action_size, dtype=tf.float32)

                log_policy = tf.log(tf.reduce_sum(
                    self.policy * action_one_hot, axis=1))
                old_log_policy = tf.log(tf.reduce_sum(
                    self.old_policy * action_one_hot, axis=1))

                ratio = tf.exp(log_policy - old_log_policy)  # pnew / pold

                std_advantages = (self.advantages - tf.reduce_mean(self.advantages)) / (tf.math.reduce_std(self.advantages) + 1e-8)
                cliped_ratio = tf.clip_by_value(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                policy_loss1 = - ratio * std_advantages
                policy_loss2 = - cliped_ratio * std_advantages

                # PPO's pessimistic surrogate (L^CLIP)
                self.policy_loss = tf.reduce_mean(tf.maximum(policy_loss1, policy_loss2))

                #self.entropy = tf.reduce_mean(self.calc_entropy(self.policy_logits))
                
                self.entropy = tf.reduce_mean(tf.reduce_sum(- self.policy * tf.log(tf.clip_by_value(self.policy, 1e-7, 1)), axis=1))
                self.total_loss = self.vf_coef * self.value_loss + \
                    self.policy_loss - self.ent_coef * self.entropy

                tf.summary.scalar('policy_loss', self.policy_loss)
                tf.summary.scalar('entropy_loss', self.entropy)
                tf.summary.scalar('total_loss', self.total_loss)
                tf.summary.scalar('episode_rewards', self.episode_rewards)
                tf.summary.scalar('max_episode_rewards', self.max_episode_rewards)

                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

                grads = tf.gradients(self.total_loss, self.local_vars)

                if self.max_grad_norm is not None:
                    grads, _ = tf.clip_by_global_norm(
                        grads, self.max_grad_norm)
                    grads = list(zip(grads, self.local_vars))

                def l2_norm(t): return tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
                for gradient, variable in grads:
                    tf.summary.histogram(
                        "gradients/" + variable.name, l2_norm(gradient))
                    tf.summary.histogram(
                        "variables/" + variable.name, l2_norm(variable))

                self.train_model_op = optimizer.apply_gradients(grads)
                self.summaries_op = tf.summary.merge_all()

        self.sess.run(tf.initialize_variables(tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)))

        if loadModel:
            self.load_weights()

    def create_sync_ops(self, from_model):
        return [tf.assign(to_var, from_var) for (to_var, from_var) in zip(
            tf.trainable_variables(self.name), tf.trainable_variables(from_model.name))]

    def calc_entropy(self, logits):
        a0 = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def save_weights(self):
        self.saver.save(self.sess, f"./{self.name}.ckpt")

    def load_weights(self):
        self.saver.restore(self.sess, f"./{self.name}.ckpt")

    def get_actions_and_values(self, state):
        return self.sess.run([self.policy, self.values], feed_dict={self.state: state})

    def get_action(self, state):
        return self.sess.run(self.policy, feed_dict={self.state: state})

    def get_value(self, state):
        return self.sess.run(self.values, feed_dict={self.state: state})

    def train(self, states, actions, rewards, advantages, old_policy, old_values):
        _, value_loss, policy_loss, entropy = self.sess.run([self.train_model_op, self.value_loss, self.policy_loss, self.entropy], {
            self.state: states, self.actions: actions, self.rewards: rewards, self.advantages: advantages, self.old_policy: old_policy, self.old_values: old_values})
        print(
            f"critic_loss:{value_loss}, actor_loss:{policy_loss}, entropy:{entropy}")

    def get_summary(self, states, actions, rewards, advantages, max_episode_rewards, episode_rewards, old_policy, old_values):
        summary = self.sess.run(self.summaries_op, feed_dict={
            self.max_episode_rewards: max_episode_rewards, self.episode_rewards: episode_rewards, self.state: states, self.actions: actions, self.rewards: rewards, self.advantages: advantages, self.old_policy: old_policy, self.old_values: old_values})
        return summary
