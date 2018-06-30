# Reinforcement Learning: Q-Learning
Simple Reinforcement Learning with Tensorflow Part 0: Q-Learning with Tables and Neural Networks

In this tutorial it was implemented a simple lookup-table version of the algorithm.

Unlike policy gradient methods, which attempt to learn functions which directly map an observation to an action, Q-Learning attempts to learn the value of being in a given state, and taking a specific action there. While both approaches ultimately allow us to take intelligent actions given a situation, the means of getting to that action differ significantly.

### Basic Theory - Reinforcement learning
Reinforcement learning is the method that will used in this tutorial to to enable an agent to determine the optimal policy for navigating the lake and obtaining the frisbee. To do this, we need a few mathematical tools. The first is a way of describing the environment, called a **Markov Decision Process (MDP)**. This is going to give us the language needed to determine our optimal action in any given cell, which is going to be determined through a function called the **action-value function**. This is going to give us a **Q-value**, which tells us the value of taking a particular action in a given cell. The action-value function is the most complex of these mathematical tools, and is derived from the **Bellman equation**.


### General Idea of the tutorial
For this tutorial we are going to be attempting to solve the FrozenLake environment from the OpenAI gym. For those unfamiliar, the OpenAI gym provides an easy way for people to experiment with their learning agents in an array of provided toy games. The FrozenLake environment consists of a 4x4 grid of blocks, each one either being the start block, the goal block, a safe frozen block, or a dangerous hole. The objective is to have an agent learn to navigate from the start to the goal without moving onto a hole. At any given time the agent can choose to move either up, down, left, or right. The catch is that there is a wind which occasionally blows the agent onto a space they didn’t choose. As such, perfect performance every time is impossible, but learning to avoid the holes and reach the goal are certainly still doable. The reward at every step is 0, except for entering the goal, which provides a reward of 1. Thus, we will need an algorithm that learns long-term expected rewards. This is exactly what Q-Learning is designed to provide.

The rules of the frozen lake environment:
- SFFF    (S: starting point, safe)
- FHFH    (F: frozen surface, safe)
- FFFH    (H: hole, fall to your doom)
- HFFG    (G: goal, where the frisbee is located)

In it’s simplest implementation, Q-Learning is a table of values for every state (row) and action (column) possible in the environment. Within each cell of the table, we learn a value for how good it is to take a given action within a given state. In the case of the FrozenLake environment, we have 16 possible states (one for each block), and 4 possible actions (the four directions of movement), giving us a 16x4 table of Q-values. We start by initializing the table to be uniform (all zeros), and then as we observe the rewards we obtain for various actions, we update the table accordingly.

We make updates to our Q-table using Bellman equation, which states that the expected long-term reward for a given action is equal to the immediate reward from the current action combined with the expected reward from the best future action taken at the following state. In this way, we reuse our own Q-table when estimating how to update our table for future actions.

### Resources:
1. [Introduction to Reinforcement Learning](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) - David Silver course
2. [Reinforcement Learning Demystified: Markov Decision Processe](https://towardsdatascience.com/reinforcement-learning-demystified-markov-decision-processes-part-1-bf00dda41690)
3. [The very basics of Reinforcement Learning](https://becominghuman.ai/the-very-basics-of-reinforcement-learning-154f28a79071)
4. [Reinforcement learning explained](https://www.oreilly.com/ideas/reinforcement-learning-explained)
5. [Deep Q-learning Explained](https://medium.com/@uwdlms/deep-q-learning-explained-2bd591d03f25)
