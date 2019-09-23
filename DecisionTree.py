import numpy as np
from Node import Node

class DecisionTree:

    def fit(self, X, y):
        self.root = Node(X, y)
        return self

    def predict(self, X):
        return self.root.predict(X.to_numpy())

    def print_tree(self):
        self.root.print_node()