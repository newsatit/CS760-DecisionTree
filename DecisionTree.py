import numpy as np
from Node import Node

class DecisionTree:

    def fit(self, X, y):
        self.root = Node(X, y, np.array(np.arange(len(y))))
        return self

    def predict(self, X):
        return self.root.predict(X.values)

    def print_tree(self):
        self.root.print_node()