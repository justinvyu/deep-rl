
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    @property
    def size(self):
        return len(self.memory)

    def flush(self):
        """
        Flushes the replay buffer of old experiences if the capacity is reached.
        >>> b = ReplayBuffer(5)
        >>> b.memory = [1, 2, 3, 4, 5, 6]
        >>> b.flush()
        >>> b.memory
        [2, 3, 4, 5, 6]
        """
        assert self.size >= self.capacity
        self.memory = self.memory[self.size - self.capacity:self.size]

    def add(self, experiences):
        """
        Adds a list of experiences to the replay buffer.
        >>> b = ReplayBuffer(5)
        >>> b.add([1, 2, 3, 4, 5, 6])
        >>> b.memory
        [2, 3, 4, 5, 6]
        """
        self.memory.extend(experiences)
        if self.size >= self.capacity:
            self.flush()

class DQN:
    pass