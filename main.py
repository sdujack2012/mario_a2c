import tensorflow as tf
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import time
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import pylab
from collections import deque
import random

from env_wrapper import EnvWrapper
from a2c_agent import A2CAgent
from state_generator import StateGenerator
from utils import preprocess_experiences

from training_parameters import n_env, n_steps, skip_frames, ent_coef, vf_coef, max_grad_norm, episodes_before_training, render, input_shape, lr, GAMMA, LAMBDA, load_model, frame_size, stack_size, max_steps

if __name__ == "__main__":
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    envs = [EnvWrapper(frame_size, skip_frames, stack_size)
            for i in range(n_env)]
    action_size = envs[0].get_action_size()

    train_model = A2CAgent("train_model", True, sess, input_shape, action_size,
                           lr, GAMMA, LAMBDA, max_grad_norm, ent_coef, vf_coef, load_model)

    step_model = A2CAgent("step_model", False, sess, input_shape, action_size,
                          lr, GAMMA, LAMBDA, max_grad_norm, ent_coef, vf_coef, False)
    step_model.sync_from_model(train_model)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())


    for env in envs:
        env.set_agent(step_model)

    t = 0
    while True:
        t += 1
        for i in range(n_steps):
            for env in envs:
                env.step()
        states = []
        actions = []
        cumulative_rewards = []

        for i in range(n_steps):
            for env in envs:
                tempState, tempActions, tempCumulative_rewards = preprocess_experiences(
                    env.get_experiences(), GAMMA)
                states += tempState
                actions += tempActions
                cumulative_rewards += tempCumulative_rewards
        experiences = list(zip(states, actions, cumulative_rewards))
        random.shuffle(experiences)

        states = np.array([experience[0] for experience in  experiences])
        actions = np.array([experience[1] for experience in  experiences])
        cumulative_rewards = np.array([experience[2] for experience in  experiences])

        train_model.train(states, actions, cumulative_rewards)
        step_model.sync_from_model(train_model)
        if t % 10 == 0:
            train_model.save_weights()
            print("saved weights")