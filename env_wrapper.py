from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import time
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np


from state_generator import StateGenerator
from training_parameters import skip_frames, ent_coef, vf_coef, max_grad_norm, episodes_before_training, render, input_shape, lr, GAMMA, LAMBDA, load_model, frame_size, stack_size, max_steps


class EnvWrapper():
    def __init__(self, frame_size, skip_frames, stack_size):
        self.env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0')
        self.env = BinarySpaceToDiscreteSpaceEnv(self.env, SIMPLE_MOVEMENT)
        self.agent = None
        self.frame_size = frame_size
        self.stack_size = stack_size
        self.action_size = self.env.action_space.n
        self.skip_frames = skip_frames
        self.render = False
        self.state_generator = StateGenerator(self.frame_size, self.stack_size)

        self.env.reset()
        raw_state, _, _, self.info = self.env.step(0)
        self.state = self.state_generator.get_stacked_frames(raw_state, True)
        self.experiences = []
        self.episode = 0
        self.episodeReward = 0
        self.maxEpisodeReward = 0
        self.current_episode_reward = 0

        self.done = False

    def step(self, n):
        for _ in range(n):
            policy, value = self.agent.get_actions_and_values(
                np.array([self.state]))
            action = np.random.choice(self.action_size, p=np.squeeze(policy))
            reward = 0

            for i in range(0, self.skip_frames):
                raw_state, frame_reward, done, info = self.env.step(action)
                if frame_reward == -15 or done:
                    self.episode += 1
                    done = True
                    reward = -15 * self.skip_frames
                    raw_state = self.env.reset()
                    self.episodeReward = self.current_episode_reward

                    if self.maxEpisodeReward < self.episodeReward:
                        self.maxEpisodeReward = self.episodeReward

                    self.current_episode_reward = 0
                    break
                else:
                    reward += frame_reward
                    reward += (5 if (info["score"] -
                                     self.info["score"]) > 0 else 0)

            reward /= 15 * self.skip_frames

            self.current_episode_reward += reward

            next_state = self.state_generator.get_stacked_frames(
                raw_state, done, self.episode % 20 == 0)

            experience = []
            experience.append(self.state)
            experience.append(action)
            experience.append(reward)
            experience.append(value)

            self.experiences.append(experience)
            self.state = next_state
            self.done = done
            self.info = info

            if self.done:
                break

    def get_experiences(self):
        if self.done:
            next_state_value = 0
        else:
            next_state_value = self.agent.get_value(np.array([self.state]))[0]

        collected_experiences = self.experiences
        self.experiences = []
        return next_state_value, collected_experiences

    def get_action_size(self):
        return self.action_size

    def set_agent(self, agent):
        self.agent = agent

    def set_render(self, render):
        self.render = render

    def get_max_and_current_episode_reward(self):
        return self.maxEpisodeReward, self.episodeReward
