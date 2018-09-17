import numpy as np
from Grid import Grid

# TESTING
def torodialDistTest():
    grid = Grid()
    print("-----------------------  TEST  -----------------------")
    print("\nAround Torus Examples:\n")
    print("Torus dimensions x={}, y={}".format(grid.x, grid.y))
    first = grid.nodes[0][0]
    second = grid.nodes[19][0]
    print('For Nodes (0,0) and (19,0) the Euklidian distance is:')
    dist = grid.dist(first, second)
    print(dist)
    assert (dist == 1)

    first = grid.nodes[1][1]
    second = grid.nodes[1][25]
    print('For Nodes (1,1) and (1,25) the Euklidian distance is:')
    dist = grid.dist(first, second)
    print(dist)
    assert (dist == 6)

    print("\nClassical distance still works:\n")

    first = grid.nodes[1][1]
    second = grid.nodes[3][3]
    print('For Nodes (1,1) and (3,3) the Euklidian distance is:')
    dist = grid.dist(first, second)
    print('{0:.2f}'.format(dist))
    print("------------------------------------------------------")


def bmuTest():
    g = Grid()
    print("-----------------------  TEST  -----------------------")
    print("BMU to [1,1,1,1] should be full of high values:")
    print(g.getBMU(np.array([1, 1, 1, 1])))
    print("------------------------------------------------------")


def consumeDataPointTest():
    print("-----------------------  TEST  -----------------------")
    g = Grid()
    datapoint = np.array([1, 1, 1, 1])
    print("Best Matching unit to datapoint {}:".format(datapoint))
    print(g.getBMU(datapoint))
    print("Consume {}".format(datapoint))
    g.consume(datapoint)
    print("Should be same coordinates, but BMU == {}".format(datapoint))
    bmu = g.getBMU(datapoint)
    assert all(bmu.vector.__eq__(datapoint)), "Test failed, BMU != datapoint after first insertion"
    print(bmu)
    print("\n")

    datapoint = np.array([0, 0, 0, 0])
    print("Best Matching unit to datapoint {}:".format(datapoint))
    print(g.getBMU(datapoint))
    print("Consume {}".format(datapoint))
    g.consume(datapoint)
    print("Should be same coordinates, and BMU approaching but not equal to {}".format(datapoint))
    print(g.getBMU(datapoint))
    print("------------------------------------------------------")


def runAll():
    torodialDistTest()
    bmuTest()
    consumeDataPointTest()

if __name__ == "__main__":
    runAll()
