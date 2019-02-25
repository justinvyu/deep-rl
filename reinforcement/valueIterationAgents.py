# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        states = self.mdp.getStates()
        for _ in range(self.iterations):
            newValues = self.values.copy()
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                if not actions:
                    continue # No reward gained from a future state, because there are no possible actions.
                else:
                    newValues[s] = max([self.computeQValueFromValues(s, a) for a in actions])
            self.values = newValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        return sum([prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState]) \
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action)])

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None
        qVals = util.Counter()
        for a in actions:
            qVals[a] = self.computeQValueFromValues(state, a)
        return qVals.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        for i in range(self.iterations):
            s = states[i % len(states)]
            actions = self.mdp.getPossibleActions(s)
            if not actions:
                continue  # No reward gained from a future state, because there are no
                          # possible actions from terminal state.
            else:
                self.values[s] = max([self.computeQValueFromValues(s, a) for a in actions])


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()

        # 1. Compute predecessors for each state. Dict [State] -> [Set of states]
        predecessors = {}
        for s in states:
            actions = self.mdp.getPossibleActions(s)
            neighbors = set()
            for a in actions:
                neighbors |= set([nextState for nextState, prob in self.mdp.getTransitionStatesAndProbs(s, a)
                                            if prob > 0])
            for n in neighbors:
                predecessors[n] = predecessors.get(n, set()).union({s})    # Update predecessors for each node that
                                                                           # the current state points to.

        # 2. Initialize Priority Queue
        pQueue = util.PriorityQueue()

        # 3. Fill up priority queue with initial values
        for s in states:
            actions = self.mdp.getPossibleActions(s)
            if self.mdp.isTerminal(s):
                continue
            diff = abs(self.values[s] - max([self.computeQValueFromValues(s, a) for a in actions]))
            pQueue.push(s, -diff)

        # 4. Iteratively update values, using priority
        for _ in range(self.iterations):
            if pQueue.isEmpty():
                return
            sToUpdate = pQueue.pop()

            # If not terminal, update the value of this node.
            sToUpdateActions = self.mdp.getPossibleActions(sToUpdate)
            if not self.mdp.isTerminal(sToUpdate):
                self.values[sToUpdate] = max([self.computeQValueFromValues(sToUpdate, a) for a in sToUpdateActions])

            for p in predecessors[sToUpdate]:
                pActions = self.mdp.getPossibleActions(p)
                if not self.mdp.isTerminal(p):
                    diff = abs(self.values[p] - max([self.computeQValueFromValues(p, a) for a in pActions]))
                    if diff > self.theta:
                        pQueue.update(p, -diff)
