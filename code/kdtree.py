import numpy as np

# x value for calculating representative sample size for random sampling of dataset
# per: https://select-statistics.co.uk/calculators/sample-size-calculator-population-proportion/#:~:text=This%20calculator%20uses%20the%20following,X%20%2B%20N%20%E2%80%93%201)%2C&text=and%20Z%CE%B1%2F2%20is,N%20is%20the%20population%20size.
SAMPLE_SIZE_X = 1800

# Implementation of nodes to place inside the KDTree, a binary tree where every LEAF node is a multi-dimensional point
# per: https://en.wikipedia.org/wiki/K-d_tree
class KDNode():
    # __init__(): constructor for KDNode, takes in point to store as data
    def __init__(self, point, left, right):
        self.point = point
        self.left = left
        self.right = right

# Implementation of KDTree data structure for partitioning of data
class KDTree:
    # __init__(): constructor for kd tree
    # input: dimensions of point in the kd tree, set of points to create the kdtree from
    def __init__(self, dimensions, dataset):
        self.dimensions = dimensions

        # create the tree from the given data (no root node yet, function returns the root node of the tree)
        self.root = self.__create_tree(dataset, 0) # start from depth = 0

    # __create_tree(): recursive function to create the tree, called during initialization of the tree
    # input: node to create the tree from, current dataset, depth of the tree
    # output: root node of the tree
    def __create_tree(self, current_dataset, depth):
        # base case(s), no more data or only one point
        if len(current_dataset) == 0:
            return None
        
        if len(current_dataset) == 1:
            return KDNode(current_dataset, None, None)

        # rotate dimensions
        dimension = depth % self.dimensions

        # get median and split datasets
        median = self.__get_median(dimension, current_dataset)

        # split dataset into less than and greater than median along dimension
        l_data, ge_data = self.__split_dataset(dimension, median, current_dataset)

        # create node
        node = KDNode(median, self.__create_tree(l_data, depth + 1), self.__create_tree(ge_data , depth + 1))
        return node

    # __split_dataset(): function to split a dataset into one less than the median along a dimension and one greater (or equal) to the median along the dimension
    # input: median, dimension, dataset:
    # output: tuple with 0-index as less than dataset and 1-index as greater/equal to dataset
    def __split_dataset(self, dimension, median, data):
        # initialize new datasets
        l_data = []
        ge_data = [] 

        # linear search through data
        for point in data:
            if point[dimension] < median[dimension]:
                l_data.append(point)
            else:
                ge_data.append(point)

        # return
        return (np.array(l_data), np.array(ge_data))

    # __get_median(): function to find the median along a certain dimension from a dataset
    # input: dimension, data
    # output: median and sorted data
    def __get_median(self, dimension, data):
        # get the number to select from the data
        sample_size = int((len(data) * SAMPLE_SIZE_X) / (len(data) + SAMPLE_SIZE_X - 1))

        # randomly select points (depending on data size) from data
        indices = np.random.choice(len(data), size = sample_size, replace = False)
        random_data = data[indices]

        # sort these points by dimension specified
        sorted_data = random_data[random_data[:, dimension].argsort(kind = 'mergesort')]

        # return approximate median
        return sorted_data[int(sample_size / 2)]
