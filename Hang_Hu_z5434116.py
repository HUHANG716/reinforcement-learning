# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:57:23 2023

@author: Francisco

Reinforcement learning

"""
import numpy as np
import matplotlib.pyplot as plt


class World(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.R = np.zeros(self.x * self.y)
        self.agentPos = 0

    def idx2xy(self, idx):
        x = int(idx / self.y)
        y = idx % self.y
        return x, y

    def xy2idx(self, x, y):
        return x * self.y + y

    def resetAgent(self, pos):
        self.agentPos = int(pos)

    def setReward(self, x, y, r):
        goalState = self.xy2idx(x, y)
        self.R[goalState] = r

    def getState(self):
        return self.agentPos

    def getReward(self):
        return self.R[self.agentPos]

    def getNumOfStates(self):
        return self.x * self.y

    def getNumOfActions(self):
        return 4

    def move(self, id):
        x_, y_ = self.idx2xy(self.agentPos)
        tmpX = x_
        tmpY = y_
        if id == 0:  # move DOWN
            tmpX += 1

        elif id == 1:  # move UP
            tmpX -= 1

        elif id == 2:  # move RIGHT
            tmpY += 1

        elif id == 3:  # move LEFT
            tmpY -= 1

        else:
            print("ERROR: Unknown action")

        if self.validMove(tmpX, tmpY):
            self.agentPos = self.xy2idx(tmpX, tmpY)

    def validMove(self, x, y):
        valid = True
        if x < 0 or x >= self.x:
            valid = False
        if y < 0 or y >= self.y:
            valid = False
        return valid


class Agent(object):
    def __init__(self, world):
        self.world = world
        self.numOfActions = self.world.getNumOfActions()
        self.numOfStates = self.world.getNumOfStates()
        self.Q = np.loadtxt("initial_Q_values.txt")
        self.rnd_nums = np.loadtxt("random_numbers.txt")
        self.rnd_nums_idx = 0
        self.alpha = 0.7
        self.gamma = 0.4
        self.epsilon = 0.25
        self.reward = []
        self.steps = []
        self.itr = 1000

    # softmax action selection
    def softmaxActionSelection(self, state, t=0.1):
        # create an array consist of [e^Q(s,0)/t, e^Q(s,1)/t, e^Q(s,2)/t, e^Q(s,3)/t...]
        actions = np.exp(self.Q[state, :] / t)

        # make the array in the range [0,1]
        prob_actions = actions / np.sum(actions)  # total possibility = 1

        # calculate the probability interval
        # if [0.1,0.2,0.3,0.4] it will become [0.1,0.3,0.6,1]
        cum_prob_actions = np.cumsum(prob_actions)
        rnd = self.get_nxt_rnd_num()
        action = np.searchsorted(cum_prob_actions, rnd)
        return action

    #
    # epsilon-greedy action selection
    def epsilonGreedyActionSelection(self, state):
        rnd = self.get_nxt_rnd_num()
        if rnd < self.epsilon:

            rnd = self.get_nxt_rnd_num()
            if rnd <= 0.25:
                action = 0  # down

            elif rnd <= 0.5:
                action = 1  # up

            elif rnd <= 0.75:
                action = 2  # right

            else:
                action = 3  # left

        else:
            action = np.argmax(self.Q[state, :])

        return action

    def get_nxt_rnd_num(self):
        rnd = self.rnd_nums[self.rnd_nums_idx]
        self.rnd_nums_idx += 1
        return rnd

    def selectAction(self, strategy, state):

        if strategy == "EG":
            return self.epsilonGreedyActionSelection(state)
        elif strategy == "SM":
            return self.softmaxActionSelection(state)
        else:
            raise ValueError("Must choose a valid action selecting strategy")

    def calculate_q_value(self, prev_Q_value, state_new, a_new, reward, method):

        if method == "QLearning":
            result = prev_Q_value + self.alpha * (
                    reward + self.gamma * np.max(self.Q[state_new, :]) - prev_Q_value
            )

        elif method == "SARSA":
            result = prev_Q_value + self.alpha * (reward +
                                                  self.gamma * self.Q[state_new, a_new] -
                                                  prev_Q_value)

        else:
            raise ValueError("Must choose a valid learning method")
        return result

    def train(self, iter, method, strategy):
        self.itr = iter

        for itr in range(iter):
            accumulatedR = 0.0
            state = 0

            self.world.resetAgent(state)

            # do action
            a = self.selectAction(strategy, state)

            # start the episode
            episode = True
            # initialize accumulatedR and accumulatedSteps in this episode

            accumulatedSteps = 0.0
            while episode:

                # perform action
                self.world.move(a)
                # get reward after moving
                reward = self.world.getReward()
                accumulatedR += reward
                state_new = int(self.world.getState())
                # record accumulated reward within this episode

                # record steps within this episode
                accumulatedSteps += 1

                # do new action
                a_new = self.selectAction(strategy, state_new)



                # update Q-values
                # parameters
                # 1. current Q value
                # 2. new state
                # 3. new action
                # 4. reward after moving
                # 5. learning method Q-learning or SARSA
                # update Q-values
                self.Q[state, a] = self.calculate_q_value(self.Q[state, a],
                                                          state_new,
                                                          a_new,
                                                          reward,
                                                          method)
                state = state_new
                a = a_new

                if reward == 1.0:
                    episode = False
                    self.Q[state_new, :] = 0
                    self.steps.append(accumulatedSteps)
                    self.reward.append(accumulatedR)
                    accumulatedR = 0.0

    def train_SARSA_EG(self, iter):
        self.train(iter, method="SARSA", strategy="EG")

    def train_SARSA_SM(self, iter):
        self.train(iter, method="SARSA", strategy="SM")

    def train_Q_learning_SM(self, iter):
        self.train(iter, method="QLearning", strategy="SM")

    def train_Q_learning_EG(self, iter):
        self.train(iter, "QLearning", "EG")

    def plotReward(self):

        plt.title("Accumulated Reward")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")

        episodes = np.arange(0, 1000)
        plt.grid(True)

        plt.plot(episodes, self.reward, label="Reward", color="red")
        plt.legend()
        plt.show()

    def plotSteps(self):
        plt.title("Steps per Episode")
        plt.xlabel("Episodes")
        plt.ylabel("Steps")
        episodes = np.arange(0, self.itr)

        plt.grid(True)
        plt.plot(episodes, self.steps, label="Steps", color="green")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    world = World(3, 4)
    world.setReward(2, 3, 1.0)  # Goal state
    world.setReward(1, 1, -1.0)  # Fear region

    # Q_learning with epsilon-greedy

    learner = Agent(world)
    learner.train_Q_learning_EG(1000)
    learner.plotReward()
    print(np.array(learner.reward))

    # Q_learning with softmax

    # learner = Agent(world)
    # learner.train_Q_learning_SM(1000)
    # learner.plotReward()

    # SARSA with epsilon-greedy

    # learner = Agent(world)
    # learner.train_SARSA_EG(1000)
    # learner.plotReward()

    # SARSA with softmax

    # learner = Agent(world)
    # learner.train_SARSA_SM(1000)
    # learner.plotReward()
