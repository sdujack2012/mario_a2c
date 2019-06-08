
import numpy as np

def calculate_cumulative_rewords_and_advantages(rewards, values, next_state_value, gamma):
    length = len(rewards)
    cumulative_rewords = np.zeros(length)
    advantages = np.zeros(length)

    cumul_r = next_state_value
    for t in reversed(range(length)):
        cumul_r = rewards[t] + cumul_r * gamma
        cumulative_rewords[t] = cumul_r
        advantages[t] = cumulative_rewords[t] - values[t]

    return cumulative_rewords.tolist(), advantages.tolist()

def preprocess_experiences(experiences, next_state_value, gamma):
    state_index = 0
    action_index = 1
    reward_index = 2
    value_index = 3

    states = [experience[state_index] for experience in experiences]
    actions = [experience[action_index] for experience in experiences]
    rewards = [experience[reward_index] for experience in experiences]
    values = [experience[value_index] for experience in experiences]
    cumulative_rewords, advantages = calculate_cumulative_rewords_and_advantages(rewards, values, next_state_value, gamma)

    return states, actions, cumulative_rewords, advantages