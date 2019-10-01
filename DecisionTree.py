import numpy as np
from Node import Node

class DecisionTree:

    def __init__(self, list_first_candidates=False):
        self.list_first_candidates = list_first_candidates

    def fit(self, X, Y):
        self.root = Node(X, Y, list_candidates=self.list_first_candidates)
        return self

    def count_nodes(self):
        return self.root.count_nodes()

    def predict(self, X):
        return self.root.predict(X)

    def print_tree(self):
        self.root.print_node()