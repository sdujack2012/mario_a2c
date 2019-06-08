import tensorflow as tf
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import time
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import pylab
from collections import deque
import random
from multiprocessing.pool import ThreadPool

from env_wrapper import EnvWrapper
from a2c_agent import A2CAgent
from state_generator import StateGenerator
from utils import preprocess_experiences

from training_parameters import n_env, n_steps, skip_frames, ent_coef, vf_coef, max_grad_norm, episodes_before_training, render, input_shape, lr, GAMMA, LAMBDA, load_model, frame_size, stack_size, max_steps
if __name__ == "__main__":
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    action_size = env.action_space.n

    # envs[0].set_render(True)

    train_model = A2CAgent("train_model", True, sess, input_shape, action_size,
                           lr, GAMMA, LAMBDA, max_grad_norm, ent_coef, vf_coef, True)
    sess.run(tf.global_variables_initializer())
    while True:
        state_generator = StateGenerator(frame_size, stack_size)
        state = state_generator.get_stacked_frames(env.reset(), True)

        episodes_reward = 0
        while True:
            env.render()

            policy, value = train_model.get_actions_and_values(np.array([state]))
            action = np.random.choice(
                np.arange(action_size), 1, p=np.squeeze(policy))[0]
            env.step(action)
            
            reward = 0
            for i in range(0, skip_frames):
                raw_state, frame_reward, done, info = env.step(action)
                if frame_reward == -15:
                    done = True
                    reward = -15
                    raw_state = env.reset()
                    break
                else:
                    reward += frame_reward

            reward /= 60
            episodes_reward += reward
            state = state_generator.get_stacked_frames(raw_state, False)
        print("score:", episodes_reward)