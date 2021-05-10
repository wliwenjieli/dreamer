import util
import numpy as np
from qlearningAgents import PacmanQAgent

class ExperienceReplay(PacmanQAgent):
    def __init__(self, weights=None):
        # read weights from previous training (awake) session
        if weights != None:
            self.weights = weights
        else:
            self.weights = util.Counter()

    def update(self, state, action, reward, nextState):
        """
           Should update your weights based on transition
        """
        "next value"
        nextActions = self.getLegalActions(nextState)
        nextQ = util.Counter()
        nextValue = 0
        if nextActions:
            for a in nextActions:
                nextQ[a] = self.getQValue(nextState, a)
            bestKey = nextQ.argMax()
            nextValue = nextQ[bestKey]

        "weight update"
        feat = self.featExtractor.getFeatures(state, action)
        for k in feat.keys():
            correction = (reward + self.discount * nextValue) - self.getQValue(state, action)
            if self.weights[k]:
                self.weights[k] += self.alpha * correction * feat[k]
            else:
                self.weights[k] = self.alpha * correction * feat[k]

        return self.weights

    def replay(self, replayBuffer, capacity):
        # randomly sample events to add to the buffer until it reaches capacity
        buffer = []
        while replayBuffer.isEmpty() == False:
            buffer.append(replayBuffer.pop())

        # randomly sample memories from the buffer
        indices = np.random.choice(len(buffer), capacity, replace=True)

        # replay
        for idx in indices:
            state, action, reward, nextState = buffer[idx].pop()
            print("next state=", nextState)
            self.weights = self.update(state, action, reward, nextState)

        return self.weights


