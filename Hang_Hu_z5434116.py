# -*- coding: utf-8 -*-
"""
Individual Assignment 1

Submitter: Hang Hu z5434116

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
    def __init__(self, world, alpha=0.7, gamma=0.4, epsilon=0.25, t=0.1):
        self.world = world
        self.numOfActions = self.world.getNumOfActions()
        self.numOfStates = self.world.getNumOfStates()
        self.Q = np.loadtxt("initial_Q_values.txt")
        self.rnd_nums = np.loadtxt("random_numbers.txt")
        self.rnd_nums_idx = 0
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.accum_reward_list = np.array([])
        self.accum_step_list = np.array([])
        self.t = t
        # default iteration
        self.itr = 1000

    # softmax action selection
    def softmaxActionSelection(self, state):
        # create an array consist of [e^Q(s,0)/t, e^Q(s,1)/t, e^Q(s,2)/t...]
        actions = np.exp(self.Q[state, :] / self.t)

        # normalization
        prob_actions = actions / np.sum(actions)  # total possibility = 1

        # transform to cumulative probability so that we can use searchsorted
        # e.g. [0.1,0.2,0.3,0.4] => [0.1,0.3,0.6,1]
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

    def train(self, itr, method, strategy):
        self.itr = itr

        for itr in range(itr):

            state = 0
            a_new = None
            self.world.resetAgent(state)
            # start the episode
            episode = True

            # choose the first action
            a = self.selectAction(strategy, state)

            # initialize accumulatedR and accumulatedSteps in this episode
            accumulatedR = 0.0
            accumulatedSteps = 0.0
            while episode:

                # perform action
                self.world.move(a)
                # get reward and new state after moving
                reward = self.world.getReward()
                state_new = int(self.world.getState())

                # record accumulated reward within this episode
                accumulatedR += reward
                # record steps within this episode
                accumulatedSteps += 1

                # do new action
                if method == "SARSA":
                    # SARSA will select new action here
                    a_new = self.selectAction(strategy, state_new)

                if method == "QLearning":
                    # do nothing
                    pass

                # update Q-values
                # 1. current Q value
                # 2. new state
                # 3. new action
                # 4. reward after moving
                # 5. learning method: Q-learning / SARSA
                self.Q[state, a] = self.calculate_q_value(self.Q[state, a],
                                                          state_new,
                                                          a_new,
                                                          reward,
                                                          method)
                # update state
                state = state_new

                # reached the goal
                if reward == 1.0:
                    episode = False
                    self.Q[state_new, :] = 0
                    self.accum_step_list = np.append(self.accum_step_list, accumulatedSteps)
                    self.accum_reward_list = np.append(self.accum_reward_list, accumulatedR)
                    accumulatedR = 0.0
                # have not reached the goal
                else:
                    if method == "SARSA":
                        # the new action is the action that will be taken in the next step
                        a = a_new
                    if method == "QLearning":
                        # select new action for Q-learning before next step
                        a = self.selectAction(strategy, state)

    def train_SARSA_EG(self, itr):
        self.train(itr, method="SARSA", strategy="EG")

    def train_SARSA_SM(self, itr):
        self.train(itr, method="SARSA", strategy="SM")

    def train_Q_learning_SM(self, itr):
        self.train(itr, method="QLearning", strategy="SM")

    def train_Q_learning_EG(self, itr):
        self.train(itr, "QLearning", "EG")

    def plotReward(self):

        plt.title("Accumulated Reward")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        episodes = np.arange(0, self.itr)
        plt.grid(True)
        plt.plot(episodes, self.accum_reward_list, label="Reward", color="red")
        plt.legend()
        plt.show()

    def plotSteps(self):
        plt.title("Steps per Episode")
        plt.xlabel("Episodes")
        plt.ylabel("Steps")
        episodes = np.arange(0, self.itr)
        plt.grid(True)
        plt.plot(episodes, self.accum_step_list, label="Steps", color="green")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    world = World(3, 4)
    world.setReward(2, 3, 1.0)  # Goal state
    world.setReward(1, 1, -1.0)  # Fear region

    # Q_learning with epsilon-greedy
    learner_Q_EG = Agent(world)
    learner_Q_EG.train_Q_learning_EG(1000)

    # Q_learning with softmax
    learner_Q_SM = Agent(world)
    learner_Q_SM.train_Q_learning_SM(1000)

    # SARSA with epsilon-greedy
    learner_S_EG = Agent(world)
    learner_S_EG.train_SARSA_EG(1000)

    # SARSA with softmax
    learner_S_SM = Agent(world)
    learner_S_SM.train_SARSA_SM(1000)

    # plot accumulated reward
    learner_Q_EG.plotReward()
    learner_Q_SM.plotReward()
    learner_S_EG.plotReward()
    learner_S_SM.plotReward()

    # plot steps per episode
    learner_Q_EG.plotSteps()
    learner_Q_SM.plotSteps()
    learner_S_EG.plotSteps()
    learner_S_SM.plotSteps()
