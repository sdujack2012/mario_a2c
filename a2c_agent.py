
import keras
from keras import backend as K
from keras.layers import Dense, Add, Flatten, Lambda, Input, Conv2D, BatchNormalization, Activation, Concatenate
from keras.models import Model
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt

class A2CAgent():
    def __init__(self, input_shape, action_size, previous_actions_size, lr, GAMMA, LAMBDA, loadModel):
        self.input_shape = input_shape
        self.action_size = action_size
        self.previous_actions_size = previous_actions_size
        self.lr = lr
        self.GAMMA = GAMMA
        self.LAMBDA = LAMBDA

        self.model, self.train_model = self.build_agent()
        if loadModel:
            self.load_weights()

    def build_agent(self):
        image_input = Input(shape=self.input_shape)

        image_output = Conv2D(
            filters=32, padding='valid', kernel_size=(8, 8), strides=4)(image_input)
        image_output = BatchNormalization(
            trainable=True)(image_output)
        image_output = Activation("elu")(image_output)

        image_output = Conv2D(
            filters=64, padding='valid', kernel_size=(4, 4), strides=2)(image_output)
        image_output = BatchNormalization(
            trainable=True)(image_output)
        image_output = Activation("elu")(image_output)

        image_output = Conv2D(
            filters=128, padding='valid', kernel_size=(3, 3), strides=1)(image_output)
        image_output = BatchNormalization(
            trainable=True)(image_output)
        image_output = Activation("elu")(image_output)

        image_output = Flatten()(image_output)

        previous_actions_input = Input(shape=(self.previous_actions_size,))

        merged_out = Concatenate()([image_output, previous_actions_input])

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

        optimizer= RMSprop(lr=self.lr, epsilon=0.1, rho=0.99)

        weighted_actions = K.sum(action_pl * model.output[0], axis=1)
        neg_log =  0 - K.log(weighted_actions)
        
        actor_loss = K.mean(neg_log * advantages_pl)

        critic_loss = K.mean(K.square(discounted_rewards_pl - model.output[1]))

        loss = critic_loss + actor_loss

        updates = optimizer.get_updates(model.trainable_weights, [], loss)
        
        train_model = K.function([model.input[0], model.input[1], action_pl, advantages_pl, discounted_rewards_pl], [critic_loss, actor_loss, loss], updates=updates)

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
        discounted_rewards = np.array(self.discount(rewards))
        advantages = np.array(self.get_gaes(rewards, state_values, next_state_values, self.GAMMA, self.LAMBDA))
        # Networks optimization
       
        return self.train_model([states, previous_actions, actions, advantages, discounted_rewards])

    def train_with_experiences(self, experiences):
        state_index = 0
        previous_action_index = 1
        action_index = 2
        state_value_index = 3
        reward_index = 4
        done_index = 5
        next_state_index = 6
        next_state_value_index = 7

        states = np.array([experience[state_index] for experience in  experiences])
        previous_actions = np.array([experience[previous_action_index] for experience in  experiences])
        state_values = np.array([experience[state_value_index] for experience in  experiences])
        next_state_values = np.array([experience[next_state_value_index] for experience in  experiences])
        actions = np.array([experience[action_index] for experience in  experiences])
        rewards = np.array([experience[reward_index] for experience in  experiences])
        dones = np.array([experience[done_index] for experience in  experiences])

        # Networks optimization
        critic_loss, actor_loss, loss = self.train(states, previous_actions, state_values, next_state_values, actions, rewards, dones)
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