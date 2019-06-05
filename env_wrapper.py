from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import time
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np


from state_generator import StateGenerator
from training_parameters import skip_frames, ent_coef, vf_coef, max_grad_norm, episodes_before_training, render, input_shape, lr, GAMMA, LAMBDA, load_model, frame_size, stack_size, max_steps


class EnvWrapper():
    def __init__(self, frame_size, skip_frames, stack_size):
        self.env = gym_super_mario_bros.make('SuperMarioBros-v0')
        self.env = BinarySpaceToDiscreteSpaceEnv(self.env, SIMPLE_MOVEMENT)
        self.agent = None
        self.frame_size = frame_size
        self.stack_size = stack_size
        self.action_size = self.env.action_space.n
        self.skip_frames = skip_frames

        self.state_generator = StateGenerator(self.frame_size, self.stack_size)

        self.state = self.state_generator.get_stacked_frames(
            self.env.reset(), True)

        self.experiences = []

    def step(self):
        policy, state_value = self.agent.get_action_and_value(
            np.array([self.state]))
        action = np.random.choice(
            np.arange(self.action_size), 1, p=np.squeeze(policy))[0]

        reward = 0
        for i in range(0, self.skip_frames):
            raw_state, frame_reward, done, info = self.env.step(action)
            if frame_reward == -15:
                done = True
                reward = -15
                raw_state = self.env.reset()
                break
            else:
                reward += frame_reward

        reward /= 15

        next_state = self.state_generator.get_stacked_frames(raw_state, done)

        experience = []
        experience.append(self.state)
        experience.append(action)
        experience.append(reward)
        experience.append(state_value)
        experience.append(done)

        self.experiences.append(experience)
        self.state = next_state

    def get_experiences(self):
        collected_experiences = self.experiences
        self.experiences = []
        return collected_experiences

    def get_action_size(self):
        return self.action_size
    
    def set_agent(self, agent):
        self.agent = agent
