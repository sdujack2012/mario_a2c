
import numpy as np

def preprocess_experiences(rewards, values, next_values, dones, GAMMA, LAMBDA):

    length = len(rewards)

    rewards = np.array(rewards)
    values = np.array(values)
    next_values = np.array(next_values)
    dones = np.array(dones)

    delta = rewards + next_values * GAMMA * (1 - dones) - values
    
    gaes = np.zeros(length) #gae

    curr_gae = 0
    for t in reversed(range(length)):
        curr_gae = gaes[t] = delta[t] + GAMMA * LAMBDA * curr_gae * (1 - dones[t])

    future_rewords = values + gaes

    return gaes.tolist(), future_rewords.tolist()