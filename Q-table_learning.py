"""
REINFORCEMENT LEARNING Q-TABLE

Tutorial based on the OpenAI gym's FrozenLake environment
(https://gym.openai.com/envs/FrozenLake-v0/)

The goal of this program is to get a frisbee located on a frozen lake.

The environment is build on the basis of a grid with dimensions 4x4:
    S  F  F  F                               where:  S: starting point, safe
    F  H  F  H                                       F: frozen surface, safe
    F  F  F  H                                       H: hole, fall to your doom
    H  F  F  G                                       G: goal, where the frisbee is located
"""


import gym
import numpy as np


# load the environment
env = gym.make('FrozenLake-v0')


# The size of the table is 16x4: 16 observable states and 4 possible actions.
# The states are in this case rows of the Q-table, while actions are define by columns.
# Each entry defines the reward for perfoming the action in the column, at the location in the row.
#
# So the Q-table looks like this:
#              U   R   D   L
# 1  (1,1) S:  x   x   x   x
# 2  (1,2) F:  x   x   x   x
# 3  (1,3) F:  x   x   x   x
# 4  (1,4) F:  x   x   x   x
# 5  (2,1) F:  x   x   x   x
# 6  (2,2) H:  x   x   x   x
# 7  (2,3) F:  x   x   x   x
# 8  (2,4) H:  x   x   x   x
# 9  (3,1) F:  x   x   x   x
# 10 (3,2) F:  x   x   x   x
# 11 (3,3) F:  x   x   x   x
# 12 (3,4) H:  x   x   x   x
# 13 (4,1) H:  x   x   x   x
# 14 (4,2) F:  x   x   x   x
# 15 (4,3) F:  x   x   x   x
# 16 (4,4) G:  x   x   x   x


# ============================== IMPLEMENT Q-TABLE LEARNING ALGORITHM ==============================

# Initialize table with all zeros - matrix 16 x 4
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Set learning parameters
learning_rate = 0.8  # defines how quickly/slowly the agent learns
gamma = 0.95  # define how short/long-term oriented is agent
num_epochs = 2000  # define time in which "learnig" will be performed
steps = 99  # the maximum number steps to take

# create list to contain total reward per episode
rList = []


for i in range(num_epochs):

    # initialise: reset environment and get first new observation
    s = env.reset()  # reset the gym environment and obtain the initial state
    rAll = 0  # total sum of rewards - initialised to zero.
    done = False  # boolean initialisation instructing whether the episode has terminated or not
    j = 0  # initialised number of steps taken before termination of the episode
    path = []  # initialise the path, which will show all steps

    # ========================= Q-TABLE LEARNING ALGORITHM ========================================
    while j < steps:
        j += 1  # increase the number of steps by one

        # Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n)*(1. / (i + 1)))
        # np.argmax(Q[s,:])  - chooses the action which corresponds to the maximum Q-value
        #                      in the Q-table
        # np.random.randn(1,env.action_space.n) - generates the noise: an array of shape
        #                                         (1, action_space) – (1, 4), filled with random
        #                                         floats sampled from a univariate “normal”
        #                                         (Gaussian) distribution, of mean 0 and variance 1
        # * (1./(i+1)) - ensures that the noise decreases as the number of episodes increases – thus
        #                over time the agent tends to be more consistent and less random in its
        #                choice of actions.

        # Get new state and reward from environment
        s1, r, d, _ = env.step(a)
        # # env.step(a) function returns the result of the action in 4 values:
        # observation (object), reward (float), done (boolean), info (dict).

        # Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + learning_rate * (r + gamma*np.max(Q[s1, :]) - Q[s, a])
        # This is based on the Bellman equation: Q(s,a) = r + γ*max(Q(s',a'))
        # The immediate reward plus the discounted present value of future rewards.
        # To update the Q-values:
        # 1. Find the difference between this new Q-value and the previous Q-value
        #    (= 0 if there's no change)
        #                       Δ = (r + γ*max(Q(s',a'))) - Q(s,a)
        # 2. Modify this difference with a learning rate (α) to ensure the agent doesn't
        #    learn "too fast"
        #                      αΔ = α[ r + γ*max(Q(s',a')) - Q(s,a) ]
        # 3. Update our Q-table values with the value we had before, plus the difference
        #         Q'(s,a) = Q(s,a) + αΔ = Q(s,a) + α[ r + *max(Q(s',a')) - Q(s,a) ]

        rAll += r  # add reward obtained in this state to the total reward
        s = s1  # set the state to the next state

        path.append(s)  # save the movement in the path

        if done is True:
            break  # exit the while loop and begin a new episode if done

        rList.append(rAll)  # list of the total reward receieved per episode


# ==================================== PRESENT THE RESULTS ========================================

# print score over time
# prints the average reward receieved for all episodes (range = [0:1])
print('Score over time: ', (sum(rList) / num_epochs))

# Print the final Q-Table values
print('Final Q-Table values: \n', Q)

# print the path
print('The final path of the agent after, ', num_epochs, 'is: \n', np.asarray(path).reshape(9, 11))
