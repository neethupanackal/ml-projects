import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.gamma = 0.9
        self.alpha = 0.9

    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if state in self.Q :
            epsolon = 1/(i_episode)
            actionProb = np.ones(self.nA) * epsolon / (self.nA-1)
            maxAction = np.argmax(self.Q[state])
            actionProb[maxAction] =  1 -  epsolon
            return actionProb, np.random.choice(self.nA , p = actionProb)
        else:
            actionProb = np.ones(self.nA) / (self.nA)
            return actionProb, np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, policyS_next, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] +=  self.alpha  *(reward + (self.gamma*np.dot(self.Q[next_state],policyS_next)) - self.Q[state][action])
        
    def step_action(self, state, action, reward, next_state,action_next,done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] +=  self.alpha  *(reward + (self.gamma*self.Q[next_state][action_next]) - self.Q[state][action])
        