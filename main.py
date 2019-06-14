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

from training_parameters import clip_range, sample_size, epoch, n_env, n_steps, skip_frames, ent_coef, vf_coef, max_grad_norm, episodes_before_training, render, input_shape, lr, GAMMA, LAMBDA, load_model, frame_size, stack_size, max_steps


def run_env(env):
    env.step(n_steps)


if __name__ == "__main__":
    n_env = multiprocessing.cpu_count()
    envs = [EnvWrapper(frame_size, skip_frames, stack_size)
            for i in range(n_env)]
    action_size = envs[0].get_action_size()

    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    train_model = A2CAgent("train_model", True, sess, input_shape, action_size,
                           lr, GAMMA, LAMBDA, max_grad_norm, ent_coef, vf_coef, clip_range, load_model)

    old_model = A2CAgent("old_model", False, sess, input_shape, action_size,
                         lr, GAMMA, LAMBDA, max_grad_norm, ent_coef, vf_coef, clip_range, False)

    sync_ops = old_model.create_sync_ops(train_model)
    sess.run(sync_ops)
    summary_writer = tf.summary.FileWriter("./log/sum", sess.graph)

    # envs[0].set_render(True)

    for env in envs:
        env.set_agent(old_model)
    p = ThreadPool(n_env)

    t = 0
    while True:
        t += 1

        p.map(run_env, envs)

        states = []
        actions = []
        future_rewords = []
        advantages = []
        policies = []
        values = []

        for env in envs:
            env_states, env_policies, env_actions, env_rewards, env_values, env_next_values, env_dones = env.get_experiences()
            env_gaes, env_future_rewords = preprocess_experiences(
                env_rewards, env_values, env_next_values, env_dones, GAMMA, LAMBDA)

            states += env_states
            policies += env_policies
            actions += env_actions
            future_rewords += env_future_rewords
            advantages += env_gaes
            values += env_values

        experiences = list(
            zip(states, actions, future_rewords, advantages, policies, values))
        len_experiences = len(experiences)
        print("experience size:", len_experiences)

        for current_epoch in range(epoch):
            print("current_epoch:", current_epoch)
            random.shuffle(experiences)

            for bacth_index in range(0, len_experiences, sample_size):
                sampled_experiences = experiences[bacth_index:bacth_index + sample_size]
                states = np.array([experience[0]
                                   for experience in sampled_experiences])
                actions = np.array([experience[1]
                                    for experience in sampled_experiences])
                future_rewords = np.array([experience[2]
                                           for experience in sampled_experiences])
                advantages = np.array([experience[3]
                                       for experience in sampled_experiences])
                policies = np.array([experience[4]
                                     for experience in sampled_experiences])
                values = np.array([experience[5]
                                   for experience in sampled_experiences])

                train_model.train(states, actions, future_rewords,
                                  advantages, policies, values)

        sess.run(sync_ops)

        if t % 10 == 0:
            print("saving weights")
            train_model.save_weights()

            print("adding summary")
            max_episode_reward, episode_reward = envs[0].get_max_and_current_episode_reward(
            )

            summary = train_model.get_summary(
                states, actions, future_rewords, advantages, max_episode_reward, episode_reward, policies, values)
            summary_writer.add_summary(summary, t)
