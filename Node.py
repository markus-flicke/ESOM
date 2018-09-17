import numpy as np

class Node:
    """
    Contains self.vector with random normalised values.
    """
    def __init__(self, x, y, length):
        self.x = x
        self.y = y
        self.vector = np.random.rand(length)

    def dist(self, vector):
        return np.linalg.norm(self.vector - vector)

    def similarity(self, vector):
        return 1/ self.dist(vector)

    # TESTING
    def similarityTest(self):
        a = Node(0, 0, 5)
        b = Node(0, 0, 5)

        a.vector = np.array([0, 1])
        b.vector = np.array([0, 6])
        assert(a.similarity(b) == 5)

    def __str__(self):
        return "({},{}): {}".format(self.x, self.y, self.vector.__str__())

    def __repr__(self):
        return self.__str__()