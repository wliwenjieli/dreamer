import util
import numpy as np
from qlearningAgents import ApproximateQAgent

class ExperienceReplay(ApproximateQAgent):
    def __init__(self, weights=None):
        ApproximateQAgent.__init__(self)

        # read weights from previous training (awake) session
        if weights != None:
            self.weights = weights
        else:
            self.weights = util.Counter()

    def replay(self, replayBuffer, capacity):
        # randomly sample events to add to the buffer until it reaches capacity
        buffer = []
        while replayBuffer.isEmpty() == False:
            buffer.append(replayBuffer.pop())

        # randomly sample memories from the buffer
        indices = np.random.choice(len(buffer), capacity, replace=True)

        # replay
        for idx in indices:
            state, action, nextState, reward = buffer[idx].pop()
            print("next state=", nextState)
            self.weights = self.update(state, action, nextState, reward)

        return self.weights


