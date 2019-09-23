import numpy as np
from Node import Node

class DecisionTree:

    def fit(self, X, Y):
        self.root = Node(X, Y)
        return self

    def predict(self, X):
        return self.root.predict(X)

    def print_tree(self):
        self.root.print_node()