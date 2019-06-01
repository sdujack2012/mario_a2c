from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
import time
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import pylab
from collections import deque

from a2c_agent import A2CAgent
from state_generator import StateGenerator
from training_parameters import render, input_shape, lr, GAMMA, LAMBDA, load_model, frame_size, stack_size, max_steps

def to_one_hot(index, size):
    one_hot = np.zeros(size)
    one_hot[index] = 1
    return one_hot

if __name__ == "__main__":
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    # get size of state and action from environment
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    previous_actions_size = action_size * 4
 
    agent_instance = A2CAgent(input_shape, action_size, previous_actions_size, lr, GAMMA, LAMBDA, load_model)
    scores, episodes = [], []

    e = 0
    while True:
        e += 1
        done = False
        score = 0
        raw_state = env.reset()

        state_generator_instance = StateGenerator(frame_size, stack_size)
        actions_stack = deque([np.zeros(action_size) for i in range(stack_size)], maxlen=stack_size)

        state = state_generator_instance.get_stacked_frames(raw_state, True)
        previous_actions = np.hstack(actions_stack)
        steps = 0  # up to 500
        experience_batch = []
        while not done and steps < max_steps:
            if render:  # if True
                env.render()
            steps += 1
            # get e greedy action
            new_experience = []
            new_experience.append(state)
            new_experience.append(previous_actions)

            policy, state_value = agent_instance.get_action_and_value(np.array([state]), np.array([previous_actions]))
            action = np.random.choice(np.arange(action_size), 1, p = policy[0])[0]
            action_one_hot = to_one_hot(action, action_size)

            raw_state, reward, done, info = env.step(action)
            
            if reward == -15:
                done = True
            
            reward /= 15

            next_state = state_generator_instance.get_stacked_frames(raw_state, False)

            actions_stack.append(action_one_hot)
            
            policy, next_state_value = agent_instance.get_action_and_value(np.array([next_state]), np.array([np.hstack(actions_stack)]))

            new_experience.append(action_one_hot)
            new_experience.append(state_value.flatten()[0])
            new_experience.append(reward)
            new_experience.append(done)
            new_experience.append(next_state)
            new_experience.append(next_state_value.flatten()[0])

            experience_batch.append(new_experience)

            state = next_state
            previous_actions = np.hstack(actions_stack)
            
            score += reward

            if done or steps >= max_steps:
                 # every episode update the target model to be same with model (donkey and carrot), carries over to next episdoe
                scores.append(score)
                episodes.append(e)
                # 'b' is type of marking for plot
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./mario.png")
                agent_instance.train_with_experiences(experience_batch)

        if e % 5 == 0:
            print("saving model")
            agent_instance.save()
