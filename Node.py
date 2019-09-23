import numpy as np
from scipy.stats import entropy


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.entropy = self.get_entropy(y)
        self.row_count = x.shape[0]
        self.col_count = x.shape[1]
        self.val = np.argmax(np.bincount(y))
        self.info_gain = 0
        self.find_varsplit()

    def find_varsplit(self):
        for i in range(self.col_count):
            self.find_better_split(i)
        if self.is_leaf:
            return
        lhs_mask = self.x.iloc[:, self.var_idx] >= self.split
        rhs_mask = self.x.iloc[:, self.var_idx] < self.split
        self.lhs = Node(self.x[lhs_mask], self.y[lhs_mask])
        self.rhs = Node(self.x[rhs_mask], self.y[rhs_mask])

    def find_better_split(self, var_idx):
        x = self.x.to_numpy()[:, var_idx]
        for r in range(self.row_count):
            lhs = x >= x[r]
            rhs = x < x[r]

            curr_gain = self.get_info_gain(lhs, rhs)
            if curr_gain > self.info_gain:
                self.var_idx = var_idx
                self.info_gain = curr_gain
                self.split = x[r]

    def get_info_gain(self, lhs, rhs):
        y = self.y.to_numpy()
        lhs_data = y[lhs]
        rhs_data = y[rhs]
        lhs_entropy = self.get_entropy(lhs_data)
        rhs_entropy = self.get_entropy(rhs_data)
        lhs_count = lhs_data.shape[0]
        rhs_count = rhs_data.shape[0]
        lhs_weight = lhs_count/self.row_count
        rhs_weight = rhs_count/self.row_count
        info_gain = self.entropy - (lhs_weight*lhs_entropy + rhs_weight*rhs_entropy)
        return info_gain

    def get_entropy(self, data):
        p_data = np.bincount(data)
        return entropy(p_data)

    @property
    def is_leaf(self):
        return self.info_gain == 0

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf:
            return self.val
        node = self.lhs if xi[self.var_idx] >= self.split else self.rhs
        return node.predict_row(xi)

    def print_node(self, level=0):
        if self.is_leaf:
            print('\t'*level, 'Leaf')
        else:
            print('\t'*level, 'x%d >= %f' % (self.var_idx, self.split))
            self.lhs.print_node(level+1)
            self.rhs.print_node(level+1)










