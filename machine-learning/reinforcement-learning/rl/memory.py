import numpy
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', "terminal"))

class Memory():
    """ Naive Memory """   
    experiences = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, *args):
        self.experiences.append(Transition(*args))        

        if len(self.experiences) > self.capacity:
            self.experiences.pop(0)

    def sample(self, n):
        n = min(n, len(self.experiences))
        return numpy.random.sample(self.experiences, n)
    
    def is_full(self):
        return len(self.experiences) >= self.capacity


class PrioritizedMemory:   # stored as ( s, a, r, s_ , t)
    """
    prioritized by error; see Schaul T. et al. - Prioritized experience replay, 2016
    From https://github.com/jaromiru/ai_examples/blob/master/open_gym/utils.py with some modifications
    """
    e = 0.1
    b = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def add(self, error, *args):                
        self.tree.add(self._compute_priority(error), Transition(*args)) 

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = numpy.random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        self.tree.update(idx, self._compute_priority(error))

    def _compute_priority(self, error):
        return (error + self.e) ** self.b  

class SumTree:
    """From https://github.com/jaromiru/ai_examples/blob/master/open_gym/utils.py"""
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros( 2*capacity - 1 )
        self.data = numpy.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])