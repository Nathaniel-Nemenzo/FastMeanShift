import time
import math
import numpy as np

# x value for calculating representative sample size for random sampling of dataset
# per: https://select-statistics.co.uk/calculators/sample-size-calculator-population-proportion/#:~:text=This%20calculator%20uses%20the%20following,X%20%2B%20N%20%E2%80%93%201)%2C&text=and%20Z%CE%B1%2F2%20is,N%20is%20the%20population%20size.
SAMPLE_SIZE_X = 1800
SPACE_COUNT = 10

# TODO: some refactoring (alot of same stuff, don't even know if index tree needed)

# Implementation of nodes to place inside the KDTree, a binary tree where every LEAF node is a multi-dimensional point # per: https://en.wikipedia.org/wiki/K-d_tree
class KDNode():
    # __init__(): constructor for KDNode, takes in point to store as data
    def __init__(self, point, left, right, count, data):
        self.point = point
        self.left = left
        self.right = right
        self.count = count # how many of that point there are (counts equals)
        self.data = data # what data this node splits (ex: root will have full dataset, depth of 1 will have ~ half of the full dataset, and so on)
        self.boundaries = self.__get_boundaries(data)

    # __get_boundaries(): function to get the boundary points for the current dataset (just the min/max along a dimension for all dimensions into one vector)
    def __get_boundaries(self, data):
        return (np.amin(data, axis = 0), np.amax(data, axis = 0))

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
            point = np.reshape(current_dataset, (self.dimensions, ))
            return KDNode(point, None, None, 1, current_dataset) # no duplicates

        # rotate dimensions
        dimension = depth % self.dimensions

        # get median and split datasets
        median = self.__get_median(dimension, current_dataset)

        # split dataset into less than and greater than median along dimension
        l_data, ge_data, duplicates = self.__split_dataset(dimension, median, current_dataset)

        # create node
        node = KDNode(median, self.__create_tree(l_data, depth + 1), self.__create_tree(ge_data , depth + 1), duplicates, current_dataset)
        return node

    # __split_dataset(): function to split a dataset into one less than the median along a dimension and one greater (or equal) to the median along the dimension
    # input: median, dimension, dataset:
    # output: tuple with 0-index as less than dataset and 1-index as greater/equal to dataset, also the number of duplicate medians found
    def __split_dataset(self, dimension, median, data):
        # get medians that are less along axis
        l_boolvec = data[:, dimension] < median[dimension]

        # count true elements
        count = np.count_nonzero(data == median, axis = 1)

        # return less than, greater or equal to (without median duplicates), and number of median duplicates
        return (data[l_boolvec].copy(), data[np.logical_and(~l_boolvec, count != self.dimensions)].copy(), np.count_nonzero(count == self.dimensions))

    # __get_median(): function to find the median along a certain dimension from a dataset
    # input: dimension, data
    # output: median and sorted data
    def __get_median(self, dimension, data):
        # get the number to select from the data
        sample_size = math.ceil((len(data) * SAMPLE_SIZE_X) / (len(data) + SAMPLE_SIZE_X - 1))

        # randomly select points (depending on data size) from data
        indices = np.random.choice(len(data), size = sample_size, replace = False)
        random_data = data[indices.astype(int)]

        # sort these points by dimension specified
        sorted_data = random_data[random_data[:, dimension].argsort(kind = 'mergesort')]

        # return approximate median
        return sorted_data[int(sample_size / 2)]

    # __print_tree_helper(): helper function to print tree representation
    def __print_tree_helper(self, node, space):
        # base case: no node
        if node != None:
            # go right first
            self.__print_tree_helper(node.right, space + SPACE_COUNT)

            # print current node
            print("\n")
            for i in range(0, space):
                print(end = " ") 
            print(node.point, node.count)
            self.__print_tree_helper(node.left, space + SPACE_COUNT) # same level

    # print_tree(): function to print the kdtree using an in-order traversal in 2d
    def print_tree(self):
        self.__print_tree_helper(self.root, 0) # no space at first

# -------------------------------------------------------------------------------------------------
# Node for KDTreeIndices
class KDIndexNode():
    # __init__(): constructor for KDNode, takes in point to store as data
    def __init__(self, left, right, count, indices, median_indices, dataset):
        self.left = left
        self.right = right
        self.count = count # how many of that point there are (counts equals)
        self.indices = indices # what data this holds
        self.median_indices = median_indices # what median indices this data holds
        self.boundaries = self.__get_boundaries(dataset[indices])

    # __get_boundaries(): function to get the boundary points for the current dataset (just the min/max along a dimension for all dimensions into one vector)
    def __get_boundaries(self, data):
        return (np.amin(data, axis = 0), np.amax(data, axis = 0))

# Implementation of KDTree data structure for partitioning of data by index instead of actual data
class KDTreeIndices:
    # __init__(): constructor for kd tree
    # input: dimensions of point in the kd tree, set of points to create the kdtree from
    def __init__(self, dimensions, dataset):
        self.dimensions = dimensions
        self.dataset = dataset

        indices = np.where([True] * len(self.dataset))[0]
        self.dataset_with_indices = np.vstack([dataset.T, indices]).T

        self.root = self.__create_tree(np.where([True] * len(self.dataset))[0], 0) # start from depth = 0

    # __create_tree(): recursive function to create the tree, called during initialization of the tree
    # input: node to create the tree from, current dataset, depth of the tree
    # output: root node of the tree
    def __create_tree(self, current_indices, depth):
        if len(current_indices) == 0:
            return None
        
        if len(current_indices) == 1:
            return KDIndexNode(left = None, right = None, 
                    count = 1, indices = current_indices, median_indices = current_indices, dataset = self.dataset) # no duplicates

        dimension = depth % self.dimensions
        median_indices = self.__get_median_indices(dimension, current_indices)
        l_indices, ge_indices, duplicates = self.__split_dataset(dimension, median_indices, current_indices, depth)
        node = KDIndexNode(self.__create_tree(l_indices, depth + 1), self.__create_tree(ge_indices , depth + 1), duplicates, current_indices, median_indices, self.dataset)
        return node

    # __split_dataset(): function to split a dataset into one less than the median along a dimension and one greater (or equal) to the median along the dimension
    # input: median, dimension, dataset:
    # output: tuple with 0-index as less than dataset and 1-index as greater/equal to dataset, also the number of duplicate medians found
    def __split_dataset(self, dimension, median_indices, indices, depth):
        data = self.dataset[indices]
        median = np.reshape(self.dataset[median_indices[0]], (self.dimensions, ))
        l_boolvec = data[:, dimension] < median[dimension]
        count = np.count_nonzero(data == median, axis = 1)
        return (indices[np.where(l_boolvec)[0]], indices[np.where(np.logical_and(~l_boolvec, count != self.dimensions))[0]], np.count_nonzero(count == self.dimensions))

    # __get_median_indices(): function to find the index of a median along a certain dimension from a dataset
    # input: dimension, current considered indices
    # output: median index in frame of original dataset, but only w/ current considered indices
    def __get_median_indices(self, dimension, current_indices): # get the number to select from the data
        sample_size = math.ceil((len(current_indices) * SAMPLE_SIZE_X) / (len(current_indices) + SAMPLE_SIZE_X - 1))
        indices = np.random.choice(len(current_indices), size = sample_size, replace = False)
        random_data = self.dataset[current_indices][indices.astype(int)] 
        sorted_indices = random_data[:, dimension].argsort(kind = 'mergesort')

        # get counts
        counts = np.count_nonzero(random_data[sorted_indices[int(sample_size / 2)]] == self.dataset, axis = 1)
        return np.where(counts == self.dimensions)[0]

    # __print_tree_helper(): helper function to print tree representation
    def __print_tree_helper(self, node, space):
        if node != None:
            self.__print_tree_helper(node.right, space + SPACE_COUNT)
            print("\n")
            for i in range(0, space):
                print(end = " ") 
            print("median_indices: ", node.median_indices, "indices: ", node.indices, "count: ", node.count)
            self.__print_tree_helper(node.left, space + SPACE_COUNT) # same level

    # print_tree(): function to print the kdtree using an in-order traversal in 2d
    def print_tree(self):
        self.__print_tree_helper(self.root, 0) # no space at first
