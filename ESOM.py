from Node import Node
import pandas as pd
from PIL import Image
import numpy as np
import seaborn as sns


class ESOM:
    """
    ESOM implementation after hearing Knowledge Discovery at Marburg University
    Currently the __init__ consumes all data and plots an ESOM using Euklidian distance
    """

    def __init__(self, trainingDF, categoriesVector, x=5, y=10, generations = 1, frameSize = 10):
        self.categoriesVector = categoriesVector
        self.dataTraining = trainingDF
        self.x = x
        self.y = y
        self.nodes = np.array([[Node(i, j, trainingDF.shape[1]) for j in range(y)] for i in range(x)])
        self.timeWeight = 1
        self.maxDistance = self.__dist(Node(0, 0, 0), Node(self.x / 2, self.y / 2, 0))
        self.generations = generations

        # May be called outside __init__ in the future
        self.consume()
        self.plot(frameSize = frameSize)
        #

    def __dist(self, nodeA, nodeB):
        def toroidalDistanceAlongAxis(a, b, axisLength):
            straightDistance = abs(a - b)
            aroundTorusDistance = abs(abs(a - b) - axisLength)
            return min(straightDistance, aroundTorusDistance)

        dx = toroidalDistanceAlongAxis(nodeA.x, nodeB.x, self.x)
        dy = toroidalDistanceAlongAxis(nodeA.y, nodeB.y, self.y)
        return (dx ** 2 + dy ** 2) ** 0.5

    def consume(self):
        for j in range(self.generations):
            for i in self.dataTraining.index:
                self.__consumeVector(self.dataTraining.iloc[i].values)
            print("Finished generation {} of {} ({:0.0f}%)".format(j + 1, self.generations, 100*(j + 1)/self.generations))

    def __consumeVector(self, vector):
        radiusOfInfluence = 1

        def propagateNeighbourhood(bmu, vector):
            for row in self.nodes:
                for node in row:
                    distBMU = self.__dist(bmu, node) / self.maxDistance  # 0 to 1
                    if distBMU < radiusOfInfluence:
                        distWeight = (1 - distBMU) ** 4  # Cone neighbourhood function of unlimited radius of influence
                        node.vector += (vector - node.vector) * distWeight * self.timeWeight

        assert len(vector) == self.dataTraining.shape[1], "Vector doesnt have expected dimensions"
        bmu = self.__getBMU(vector)
        propagateNeighbourhood(bmu, vector)
        self.timeWeight *= 0.98  # Deceasing the time weight at every datapoint

    def __getBMU(self, vector):
        bmuSimilarity = -99999
        bmu = None
        for row in self.nodes:
            for node in row:
                if node.similarity(vector) > bmuSimilarity:
                    bmu = node
                    bmuSimilarity = bmu.similarity(vector)
        assert bmu is not None, "no best matching unit found"
        return bmu

    def plot(self, ncolors=20, frameSize=10):
        xDim = self.x * frameSize
        yDim = self.y * frameSize
        palette = sns.color_palette("RdBu", n_colors=ncolors)
        palette = list(map(lambda x: tuple(int(i * 256) for i in x), palette))  # convert to RGB from fractional
        img = Image.new('RGB', (xDim, yDim), "black")
        pixels = img.load()

        def color(pixels, x, y, length, color):
            for i in range(length):
                for j in range(length):
                    pixels[int(x * length + i), int(y * length + j)] = color

        # Normalise Similarity values

        # similarities = pd.Series([node.similarity([1,1,1,1]) for node in self.nodes.flatten()])
        # min = similarities.min()
        # max = similarities.max()

        def nearestNeighbourDistance(node):
            def getXTorusCoordinate(x):
                return (x + self.x) % self.x

            def getYTorusCoordinate(y):
                return (y + self.y) % self.y

            x = node.x
            y = node.y
            totalDistance = 0
            totalDistance += node.dist(self.nodes[getXTorusCoordinate(x + 1)][getYTorusCoordinate(y + 1)].vector)
            totalDistance += node.dist(self.nodes[getXTorusCoordinate(x + 1)][getYTorusCoordinate(y)].vector)
            totalDistance += node.dist(self.nodes[getXTorusCoordinate(x + 1)][getYTorusCoordinate(y - 1)].vector)
            totalDistance += node.dist(self.nodes[getXTorusCoordinate(x)][getYTorusCoordinate(y + 1)].vector)
            totalDistance += node.dist(self.nodes[getXTorusCoordinate(x)][getYTorusCoordinate(y - 1)].vector)
            totalDistance += node.dist(self.nodes[getXTorusCoordinate(x - 1)][getYTorusCoordinate(y + 1)].vector)
            totalDistance += node.dist(self.nodes[getXTorusCoordinate(x - 1)][getYTorusCoordinate(y)].vector)
            totalDistance += node.dist(self.nodes[getXTorusCoordinate(x - 1)][getYTorusCoordinate(y - 1)].vector)
            return totalDistance / 8

        nearestNeighbourDistances = pd.Series([nearestNeighbourDistance(node) for node in self.nodes.flatten()])
        min = nearestNeighbourDistances.min()
        max = nearestNeighbourDistances.max()

        for i, row in enumerate(self.nodes):
            for j, node in enumerate(row):
                nodeColorIDX = int(((nearestNeighbourDistance(node) - min) / (max - min)) * (ncolors - 1))
                # Nearest neighbour hyperspace distance

                assert nodeColorIDX >= 0 and nodeColorIDX < 20, nodeColorIDX
                nodeColorIDX = ncolors - 1 - nodeColorIDX
                try:
                    color(pixels, i, j, frameSize, palette[nodeColorIDX])
                except:
                    print(nodeColorIDX)
                    raise

        def drawBMU(pixels, x, y, color):
            for i in range(frameSize):
                pixels[x * frameSize + i, y * frameSize + i] = color
                pixels[x * frameSize + i, y * frameSize + frameSize - 1 - i] = color

        def drawBMUs():
            for i in self.dataTraining.index:
                species = self.categoriesVector[i]
                color = (255, 0, 0) if species == 'setosa' \
                    else (0, 255, 0) if species == 'virginica' \
                    else (0, 0, 255) if species == "versicolor" \
                    else None

                node = self.__getBMU(self.dataTraining.iloc[i].values)
                drawBMU(pixels, node.x, node.y, color)

        drawBMUs()
        img.show()

    def __str__(self):
        res = ''
        count = 0
        for row in self.nodes:
            for node in row:
                res = "{}{}: {}\n".format(res, count, node.__str__())
                count += 1
        return res

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    dataAll = pd.read_csv('iris.csv')
    dataTrain = dataAll.filter(["sepal_length", "sepal_width", "petal_length", "petal_width"])

    def normalise(df):
        for col in df.columns:
            df[col] -= df[col].min()
            df[col] /= df[col].max()
        return df

    normalised = normalise(dataTrain)
    ESOM(normalised, dataAll['species'])