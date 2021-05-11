import util
import numpy as np
from qlearningAgents import ApproximateQAgent
import pickle

class ExperienceReplay(ApproximateQAgent):
    def __init__(self, mismatch=1.0, **args):
        ApproximateQAgent.__init__(self, **args)

        # mismatch parameter: the probability that a ghost is bad in replay
        self.mismatch = mismatch

    def replay(self, replayBuffer, capacity):
        # randomly sample events to add to the buffer until it reaches capacity
        buffer = []

        if replayBuffer.isEmpty():
            print("Pac-Man does not have fear memory")
            return

        while not replayBuffer.isEmpty():
            # replayBuffer pops a queue of n events
            buffer.append(replayBuffer.pop())

        # randomly sample memories from the buffer
        indices = np.random.choice(len(buffer), capacity, replace=True)

        # replay
        for idx in indices:
            # we don't want to erase the buffer since it could be re-sampled
            tmpBuffer = buffer[idx].copy2stack()
            while not tmpBuffer.isEmpty():
                # each buffer[idx] contains one event; an event has n episodes
                state, action, nextState, reward = tmpBuffer.pop()

                # mismatch step: under with prob(mismatch) ghost doesn't attack at the end of the event
                if tmpBuffer.isEmpty(): # this is the last episode of an event
                    if np.random.random() > self.mismatch:
                        reward = 0

                # update weights
                self.weights = self.update(state, action, nextState, reward)

        f = open(self.checkpoint, "wb")
        pickle.dump(self.weights, f)
        print(self.weights)
        print('---------')
        f.close()

        return self.weights


