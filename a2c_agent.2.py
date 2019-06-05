
import tensorflow as tf
from skimage import transform 

import numpy as np
import matplotlib.pyplot as plt
import random
class A2CAgent():
    def __init__(self, input_shape, action_size, previous_actions_size, lr, GAMMA, LAMBDA, loadModel):
        self.input_shape = input_shape
        self.action_size = action_size
        self.previous_actions_size = previous_actions_size
        self.lr = lr
        self.GAMMA = GAMMA
        self.LAMBDA = LAMBDA

        self.model, self.train_model = self.build_agent()
        self.old_model = self.build_agent()[0]
        if loadModel:
            self.load_weights()

        self.old_model.set_weights(self.model.get_weights())

    def update_old_model(self):
        self.old_model.set_weights(self.model.get_weights())
        
    def build_agent(self):
        image_input = Input(shape=self.input_shape)

        image_output = Conv2D(
            filters=32, padding='valid', kernel_size=3, strides=2, kernel_initializer=keras.initializers.he_uniform(),
                                 bias_initializer=keras.initializers.he_uniform())(image_input)
        image_output = BatchNormalization(
            trainable=True)(image_output)
        image_output = Activation("elu")(image_output)

        image_output = Conv2D(
            filters=32, padding='valid', kernel_size=3, strides=2, kernel_initializer=keras.initializers.he_uniform(),
                                 bias_initializer=keras.initializers.he_uniform())(image_output)
        image_output = BatchNormalization(
            trainable=True)(image_output)
        image_output = Activation("elu")(image_output)

        image_output = Conv2D(
            filters=32, padding='valid', kernel_size=3, strides=1, kernel_initializer=keras.initializers.he_uniform(),
                                 bias_initializer=keras.initializers.he_uniform())(image_output)
        image_output = BatchNormalization(
            trainable=True)(image_output)
        image_output = Activation("elu")(image_output)
        
        image_output = Conv2D(
            filters=32, padding='valid', kernel_size=3, strides=1, kernel_initializer=keras.initializers.he_uniform(),
                                 bias_initializer=keras.initializers.he_uniform())(image_output)
        image_output = BatchNormalization(
            trainable=True)(image_output)

        image_output = Flatten()(image_output)

        previous_actions_input = Input(shape=(self.previous_actions_size,))

        merged_out = Concatenate()([image_output, previous_actions_input])
        merged_out = Activation("elu")(merged_out)

        actor_output = Dense(512)(merged_out)
        actor_output = BatchNormalization(
            trainable=True)(actor_output)
        actor_output = Activation("elu")(actor_output)
        actor_output = Dense(self.action_size, activation="softmax", kernel_initializer=keras.initializers.he_uniform(),
                                 bias_initializer=keras.initializers.he_uniform())(actor_output)

        critic_output = Dense(512)(merged_out)
        critic_output = BatchNormalization(
            trainable=True)(critic_output)
        critic_output = Activation("elu")(critic_output)
        critic_output = Dense(1, kernel_initializer=keras.initializers.he_uniform(),
                                   bias_initializer=keras.initializers.he_uniform())(critic_output)

        model = Model(
            inputs=[image_input, previous_actions_input], outputs=[actor_output, critic_output])

        action_pl = K.placeholder(shape=(None, self.action_size))
        advantages_pl = K.placeholder(shape=(None,))
        discounted_rewards_pl = K.placeholder(shape=(None,))
        old_model_output_pl = K.placeholder(shape=(None, self.action_size))

        optimizer= RMSprop(lr=self.lr, epsilon=0.1, rho=0.99)

        weighted_actions = K.sum(action_pl * model.output[0], axis=1)
        old_weighted_actions = K.sum(action_pl * old_model_output_pl, axis=1)

        ration = weighted_actions / old_weighted_actions

        cliped_ration = K.clip(ration, 0.8, 1.2)
        
        entropy = K.mean(model.output[0] * K.log(K.clip(model.output[0], 1e-10, 1)))

        actor_loss = 0 - K.mean(K.minimum(ration * advantages_pl, cliped_ration * advantages_pl) )

        critic_loss = 0.5 * K.mean(K.square(discounted_rewards_pl - model.output[1]))

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        updates = optimizer.get_updates(model.trainable_weights, [], loss)
        
        train_model = K.function([model.input[0], model.input[1], action_pl, advantages_pl, discounted_rewards_pl, old_model_output_pl], [critic_loss, actor_loss, loss], updates=updates)

        return model, train_model

    def save(self):
        self.model.save_weights('./mario_model.h5')
        self.model.save_weights('./backup_mario_model.h5')

    def load_weights(self):
        self.model.load_weights('./mario_model.h5')

    def get_action_and_value(self, states, previous_actions):
        return self.model.predict([states, previous_actions])

    def train(self, states, previous_actions, state_values, next_state_values, actions, rewards, dones):
        """ Update actor and critic networks from experience
        """
        # Compute discounted rewards and Advantage (TD. Error)
        old_model_output = self.old_model.predict([states, previous_actions])[0]
        discounted_rewards = np.array(self.discount(rewards))
        advantages = np.array(self.get_gaes(rewards, state_values, next_state_values, self.GAMMA, self.LAMBDA))

        # Networks optimization
       
        return self.train_model([states, previous_actions, actions, advantages, discounted_rewards, old_model_output])

    def process_episode_experiences(self, experiences):
        state_index = 0
        previous_action_index = 1
        action_index = 2
        state_value_index = 3
        reward_index = 4
        done_index = 5

        states = [experience[state_index] for experience in  experiences]
        previous_actions = [experience[previous_action_index] for experience in  experiences]
        state_values = [experience[state_value_index] for experience in  experiences]
        next_state_values = state_values[1:] + [0]

        actions = [experience[action_index] for experience in  experiences]
        rewards = [experience[reward_index] for experience in  experiences]
        dones = [experience[done_index] for experience in  experiences]

        return states, previous_actions, state_values, next_state_values, actions, rewards, dones

    def train_with_experiences(self, episode_experiences):
        all_states = []
        all_previous_actions = []
        all_state_values = []
        all_next_state_values = []
        all_actions = []
        all_rewards = []
        all_dones = []

        for experiences in episode_experiences:
            states, previous_actions, state_values, next_state_values, actions, rewards, dones = self.process_episode_experiences(experiences)
            all_states += states
            all_previous_actions += previous_actions
            all_state_values += state_values
            all_next_state_values += next_state_values
            all_actions += actions
            all_rewards += rewards
            all_dones += dones

        length = len(all_states)
        shuffled = list(range(length))
        random.shuffle(shuffled)
        all_states = np.array(all_states)[shuffled]
        all_previous_actions = np.array(all_previous_actions)[shuffled]
        all_state_values = np.array(all_state_values)[shuffled]
        all_next_state_values = np.array(all_next_state_values)[shuffled]
        all_actions = np.array(all_actions)[shuffled]
        all_rewards = np.array(all_rewards)[shuffled]
        all_dones = np.array(all_dones)[shuffled]

        bacth_size = 32
        bacth_index = range(0, len(all_states))
        bacthes = np.array_split(bacth_index, bacth_size)
        for batch in bacthes:
            critic_loss, actor_loss, loss = self.train(all_states[batch], all_previous_actions[batch], all_state_values[batch], all_next_state_values[batch], all_actions[batch], all_rewards[batch], all_dones[batch])
            print(f"critic_loss:{critic_loss}, actor_loss:{actor_loss}, loss:{loss}")
        
    def discount(self, r):
        discounted_r = np.zeros(len(r))
        cumul_r = 0
        for t in reversed(range(len(r))):
            cumul_r = r[t] + cumul_r * self.GAMMA
            discounted_r[t] = cumul_r
        return discounted_r
        
    # We are defining the function to get the Generalized Advantage Estimation
    def get_gaes(self, rewards, state_values, next_state_values, GAMMA, LAMBDA):
        gaes = np.array([r_t + GAMMA * next_v - v for r_t, next_v, v in zip(rewards, next_state_values, state_values)])
        for t in reversed(range(len(gaes) - 1)):
            gaes[t] = gaes[t] + LAMBDA * GAMMA * gaes[t + 1]
        return gaes