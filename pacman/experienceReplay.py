import util
import numpy as np
from qlearningAgents import ApproximateQAgent

class ExperienceReplay(ApproximateQAgent):
    def __init__(self, mismatch=1.0, weights=None):
        ApproximateQAgent.__init__(self)

        # mismatch parameter: the probability that a ghost is bad in replay
        self.mismatch = mismatch

        # read weights from previous training (awake) session
        if weights != None:
            self.weights = weights
        else:
            self.weights = util.Counter()

    def replay(self, replayBuffer, capacity):
        # randomly sample events to add to the buffer until it reaches capacity
        buffer = []
        while not replayBuffer.isEmpty():
            # replayBuffer pops a queue of n events
            buffer.append(replayBuffer.pop())

        # randomly sample memories from the buffer
        indices = np.random.choice(len(buffer), capacity, replace=True)

        # replay
        for idx in indices:
            # we don't want to erase the buffer since it could be resampled
            tmpBuffer = buffer[idx].copy2stack()
            while not tmpBuffer.isEmpty():
                state, action, nextState, reward = tmpBuffer.pop()
                self.weights = self.update(state, action, nextState, reward)

        return self.weights


