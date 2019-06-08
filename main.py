import tensorflow as tf
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import time
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import pylab
from collections import deque
import random
import multiprocessing
from multiprocessing.pool import ThreadPool

from env_wrapper import EnvWrapper
from a2c_agent import A2CAgent
from state_generator import StateGenerator
from utils import preprocess_experiences

from training_parameters import n_env, n_steps, skip_frames, ent_coef, vf_coef, max_grad_norm, episodes_before_training, render, input_shape, lr, GAMMA, LAMBDA, load_model, frame_size, stack_size, max_steps

def run_env(env):
    env.step(n_steps)

if __name__ == "__main__":
    n_env = multiprocessing.cpu_count()

    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    summary_writer = tf.summary.FileWriter("./log/sum", sess.graph)
  
    envs = [EnvWrapper(frame_size, skip_frames, stack_size)
            for i in range(n_env)]
    action_size = envs[0].get_action_size()

    # envs[0].set_render(True)

    train_model = A2CAgent("train_model", sess, input_shape, action_size,
                           lr, GAMMA, LAMBDA, max_grad_norm, ent_coef, vf_coef, load_model)
    for env in envs:
        env.set_agent(train_model)
    p = ThreadPool(n_env)

    t = 0
    while True:
        t += 1
        
        p.map(run_env, envs)

        states = []
        actions = []
        cumulative_rewards = []
        advantages = []

        for i in range(n_steps):
            for env in envs:
                next_state_value, expereiences = env.get_experiences()
                tempState, tempActions, tempCumulative_rewards, tempAdvantages = preprocess_experiences(expereiences, next_state_value, GAMMA)
                states += tempState
                actions += tempActions
                cumulative_rewards += tempCumulative_rewards
                advantages += tempAdvantages

        experiences = list(zip(states, actions, cumulative_rewards, advantages))
        random.shuffle(experiences)

        states = np.array([experience[0] for experience in  experiences])
        actions = np.array([experience[1] for experience in  experiences])
        cumulative_rewards = np.array([experience[2] for experience in  experiences])
        advantages = np.array([experience[3] for experience in  experiences])
        
        train_model.train(states, actions, cumulative_rewards, advantages)
        if t % 100 == 0:
            print("saving weights")
            train_model.save_weights()
            
            print("adding summary")
            max_episode_reward, episode_reward = envs[0].get_max_and_current_episode_reward()

            summary = train_model.get_summary(states, actions, cumulative_rewards, advantages, max_episode_reward, episode_reward)
            summary_writer.add_summary(summary, t)