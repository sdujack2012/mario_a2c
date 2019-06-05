
import tensorflow as tf
import numpy as np

def create_layers():
    layer_count = 0

    def conv2d(input_layer, kernel_size, filters, padding, strides=1, activation=None, trainable=True):
        nonlocal layer_count
        input_shape = input_layer.get_shape()
        input_dim = len(input_shape)

        W = tf.get_variable(f'w_{layer_count}', shape=(
            kernel_size, kernel_size, input_shape[input_dim - 1], filters), initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable, dtype=tf.float64)
        B = tf.get_variable(f'b_{layer_count}', shape=(
            filters), initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable, dtype=tf.float64)
        layer_count += 1
        # Conv2D wrapper, with bias and relu activation
        out = tf.nn.conv2d(input_layer, W, strides=[
                           1, strides, strides, 1], padding=padding)
        out = tf.nn.bias_add(out, B)
        return out if activation is None else activation(out)

    def dense(input_layer, output, activation=None, trainable=True):
        nonlocal layer_count
        W = tf.get_variable(f'w_{layer_count}', shape=(input_layer.get_shape()[
                            1], output), initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable, dtype=tf.float64)
        B = tf.get_variable(f'b_{layer_count}', shape=(
            output), initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable, dtype=tf.float64)
        layer_count += 1

        out = tf.add(tf.matmul(input_layer, W), B)
        return out if activation is None else activation(out)

    def flatten(input_layer):
        shape = int(np.prod(input_layer.get_shape()[1:]))
        out = tf.reshape(input_layer, [-1, shape])
        return out

    return conv2d, dense, flatten

def calculate_cumulative_rewords(rewards, dones, gamma):
    length = len(rewards)
    cumulative_rewords = np.zeros(length)

    cumul_r = 0
    for t in reversed(range(length)):
        if dones[t] == True:
            cumul_r = rewards[t]
        else: 
            cumul_r = rewards[t] + cumul_r * gamma

        cumulative_rewords[t] = cumul_r

    return cumulative_rewords.tolist()

def preprocess_experiences(experiences, gamma):
    state_index = 0
    action_index = 1
    reward_index = 2
    state_value_index = 3
    done_index = 4

    states = [experience[state_index] for experience in experiences]
    actions = [experience[action_index] for experience in experiences]
    rewards = [experience[reward_index] for experience in experiences]
    state_values = [experience[state_value_index] for experience in experiences]
    dones = [experience[done_index] for experience in experiences]
    cumulative_rewords = calculate_cumulative_rewords(rewards, dones, gamma)

    return states, actions, cumulative_rewords