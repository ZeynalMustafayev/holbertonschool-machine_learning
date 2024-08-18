#!/usr/bin/env python3
""" Task 6: 6. The predict function"""
import numpy as np


def left_child_add_prefix(text):
    """
    Adds a prefix to each line of the text to
    indicate it is the left child in the tree structure.

    Parameters:
    text : str
        The text to which the prefix will be added.

    Returns:
    str
        The text with the left child prefix added to each line.
    """
    lines = text.split("\n")
    new_text = "    +--" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("    |  "+x) + "\n"
    return new_text


def right_child_add_prefix(text):
    """
    Adds a prefix to each line of the text to indicate
    it is the right child in the tree structure.

    Parameters:
    text : str
        The text to which the prefix will be added.

    Returns:
    str
        The text with the right child prefix added to each line.
    """
    lines = text.split("\n")
    new_text = "    +--" + lines[0] + "\n"
    for x in lines[1:]:
        new_text += ("       " + x) + "\n"
    return new_text


class Node:
    """
    A class representing a node in a decision tree.

    Attributes:
    feature : int or None
        The feature used for splitting the data.
    threshold : float or None
        The threshold value for the split.
    left_child : Node or None
        The left child node.
    right_child : Node or None
        The right child node.
    is_leaf : bool
        Boolean indicating if the node is a leaf.
    is_root : bool
        Boolean indicating if the node is the root.
    sub_population : any
        The subset of data at this node.
    depth : int
        The depth of the node in the tree.

    Methods:
    max_depth_below():
        Calculates the maximum depth of the subtree rooted at this node.
    """

    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        """
        Initializes a Node with the given parameters.

        Parameters:
        feature : int or None, optional
            The feature used for splitting the data (default is None).
        threshold : float or None, optional
            The threshold value for the split (default is None).
        left_child : Node or None, optional
            The left child node (default is None).
        right_child : Node or None, optional
            The right child node (default is None).
        is_root : bool, optional
            Boolean indicating if the node is the root (default is False).
        depth : int, optional
            The depth of the node in the tree (default is 0).
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Calculates the maximum depth of the subtree rooted at this node.

        Returns:
        int
            The maximum depth of the subtree.
        """
        if self.is_leaf:
            return self.depth
        if self.left_child:
            left_depth = self.left_child.max_depth_below()
        else:
            left_depth = self.depth
        if self.right_child:
            right_depth = self.right_child.max_depth_below()
        else:
            right_depth = self.depth
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the number of nodes in the subtree rooted at this node.

        Parameters:
        only_leaves : bool, optional
            If True, count only the leaf nodes (default is False).

        Returns:
        int
            The number of nodes in the subtree.
        """
        if self.is_leaf:
            return 1
        if self.left_child:
            left_count = self.left_child.count_nodes_below(only_leaves)
        else:
            left_count = 0
        if self.right_child:
            right_count = self.right_child.count_nodes_below(only_leaves)
        else:
            right_count = 0
        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count

    def __str__(self):
        """
        Returns a string representation of the node and its children.

        Returns:
        str
            The string representation of the node.
        """
        if self.is_root:
            Type = "root "
        elif self.is_leaf:
            return f"-> leaf [value={self.value}]"
        else:
            Type = "-> node "
        if self.left_child:
            left_str = left_child_add_prefix(str(self.left_child))
        else:
            left_str = ""
        if self.right_child:
            right_str = right_child_add_prefix(str(self.right_child))
        else:
            right_str = ""
        return f"{Type}[feature={self.feature}, threshold=\
{self.threshold}]\n{left_str}{right_str}".rstrip()

    def get_leaves_below(self):
        """
        Returns a list of all leaves below this node.

        Returns:
        list
            The list of all leaves below this node.
        """
        if self.is_leaf:
            return [self]
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Update the bounds for the current node and propagate the
        bounds to its children.
        """
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        for child in [self.left_child, self.right_child]:
            if child:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                if child == self.left_child:
                    child.lower[self.feature] = self.threshold
                else:
                    child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child:
                child.update_bounds_below()

    def update_indicator(self):
        """
        Compute the indicator function for the current
        node based on the lower and upper bounds.
        """

        def is_large_enough(x):
            """
            Check if each individual has all its features
            greater than the lower bounds.

            Parameters:
            x : np.ndarray
                A 2D NumPy array of shape (n_individuals, n_features).

            Returns:

            np.ndarray
                A 1D NumPy array of boolean values
                indicating if each individual meets the condition.
            """
            return np.all(np.array([x[:, key] > self.lower[key]
                                    for key in self.lower.keys()]), axis=0)

        def is_small_enough(x):
            """
            Check if each individual has all its features
            less than or equal to the upper bounds.

            Parameters:
            x : np.ndarray
                A 2D NumPy array of shape (n_individuals, n_features).

            Returns:
            np.ndarray
                A 1D NumPy array of boolean values indicating
                if each individual meets the condition.
            """
            return np.all(np.array([x[:, key] <= self.upper[key]
                                    for key in self.upper.keys()]), axis=0)

        self.indicator = lambda x: \
            np.all(np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """
        Predict the class for a single individual at the node.

        Parameters:
        x : np.ndarray
            A 1D NumPy array representing the features of a single individual.

        Returns:
        int
            The predicted class for the individual.
        """
        if self.is_leaf:
            return self.value
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """
    A class representing a leaf node in a decision tree, inheriting from Node.

    Attributes:
    value : any
        The value predicted by the leaf.
    depth : int
        The depth of the leaf in the tree.

    Methods:
    max_depth_below():
        Returns the depth of the leaf.
    """

    def __init__(self, value, depth=None):
        """
        Initializes a Leaf with the given parameters.

        Parameters:
        value : any
            The value predicted by the leaf.
        depth : int, optional
            The depth of the leaf in the tree (default is None).
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of the leaf.

        Returns:
        int
            The depth of the leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the number of nodes in the subtree rooted at this leaf.

        Parameters:
        only_leaves : bool, optional
            If True, count only the leaf nodes (default is False).

        Returns:
        int
            The number of nodes in the subtree.
        """
        return 1

    def __str__(self):
        """
        Returns a string representation of the leaf node.

        Returns:
        str
            The string representation of the leaf node.
        """
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """
        Returns a list of all leaves below this leaf.

        Returns:
        list
            The list of all leaves below this leaf.
        """
        return [self]

    def update_bounds_below(self):
        """
        Placeholder function for updating the
        bounds for the current node and propagating the bounds
        to its children.
        """
        pass

    def pred(self, x):
        """
        Predict the class for a single individual at the leaf node.

        Parameters:
        x : np.ndarray
            A 1D NumPy array representing the features of a single individual.

        Returns:
        int
            The predicted class for the individual.
        """
        return self.value


class Decision_Tree():
    """
    A class representing a decision tree.

    Attributes:
    rng : numpy.random.Generator
        Random number generator for reproducibility.
    root : Node
        The root node of the tree.
    explanatory : any
        The explanatory features of the dataset.
    target : any
        The target values of the dataset.
    max_depth : int
        The maximum depth of the tree.
    min_pop : int
        The minimum population required to split a node.
    split_criterion : str
        The criterion used to split nodes.
    predict : any
        Method to predict the target value for a given set of features.

    Methods:
    depth():
        Returns the maximum depth of the tree.
    """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initializes a Decision_Tree with the given parameters.

        Parameters:
        max_depth : int, optional
            The maximum depth of the tree (default is 10).
        min_pop : int, optional
            The minimum population required to split a node (default is 1).
        seed : int, optional
            Seed for the random number generator (default is 0).
        split_criterion : str, optional
            The criterion used to split nodes (default is "random").
        root : Node or None, optional
            The root node of the tree (default is None).
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Returns the maximum depth of the tree.

        Returns:
        int
            The maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts the number of nodes in the decision tree.

        Parameters:
        only_leaves : bool, optional
            If True, count only the leaf nodes (default is False).

        Returns:
        int
            The number of nodes in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Returns a string representation of the decision tree.

        Returns:
        str
            The string representation of the decision tree.
        """
        return self.root.__str__() + "\n"

    def get_leaves(self):
        """
        Returns a list of all leaves in the tree.

        Returns:
        list
            The list of all leaves in the tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Update the bounds for the entire
        tree starting from the root node.
        """
        self.root.update_bounds_below()

    def update_predict(self):
        """
        Update the prediction function for the decision tree.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict(A):
            """
            Predict the class for each individual in the input
            array A using the decision tree.

            Parameters:
            A : np.ndarray
                A 2D NumPy array of shape (n_individuals,
                n_features), where each row
                represents an individual with its features.

            Returns:
            np.ndarray
                A 1D NumPy array of shape (n_individuals,),
                where each element is the predicted
                class for the corresponding individual in A.
            """
            predictions = np.zeros(A.shape[0], dtype=int)
            for i, x in enumerate(A):
                for leaf in leaves:
                    if leaf.indicator(np.array([x])):
                        predictions[i] = leaf.value
                        break
            return predictions
        self.predict = predict

    def pred(self, x):
        """
        Predict the class for a single individual using the decision tree.

        Parameters:
        x : np.ndarray
            A 1D NumPy array representing the features of a single individual.

        Returns:
        int
            The predicted class for the individual.
        """
        return self.root.pred(x)
