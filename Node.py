import numpy as np
from scipy.stats import entropy


def get_entropy(data):
    p_data = data.value_counts()
    return entropy(p_data)


class Node:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.entropy = get_entropy(Y)
        self.row_count = X.shape[0]
        self.col_count = X.shape[1]
        self.prediction = np.argmax(np.bincount(Y))
        self.info_gain = 0

        for i in range(self.col_count):
            self.update_split(i)
        if self.is_leaf:
            return
        lhs_mask = self.X.iloc[:, self.split_idx] >= self.split
        rhs_mask = self.X.iloc[:, self.split_idx] < self.split
        self.lhs = Node(self.X[lhs_mask], self.Y[lhs_mask])
        self.rhs = Node(self.X[rhs_mask], self.Y[rhs_mask])

    def update_split(self, split_idx):
        X_col = self.X.iloc[:, split_idx]
        for r in range(self.row_count):
            split = self.X.iloc[r, split_idx]
            lhs_idxs = X_col >= split
            rhs_idxs = X_col < split
            curr_gain = self.get_info_gain(lhs_idxs, rhs_idxs)
            if curr_gain > self.info_gain:
                self.split_idx = split_idx
                self.split = split
                self.info_gain = curr_gain

    def get_info_gain(self, lhs_idxs, rhs_idxs):
        lhs_Y = self.Y[lhs_idxs]
        rhs_Y = self.Y[rhs_idxs]
        lhs_entropy = get_entropy(lhs_Y)
        rhs_entropy = get_entropy(rhs_Y)
        lhs_count = lhs_Y.shape[0]
        rhs_count = rhs_Y.shape[0]
        lhs_weight = lhs_count/self.row_count
        rhs_weight = rhs_count/self.row_count
        info_gain = self.entropy - (lhs_weight*lhs_entropy + rhs_weight*rhs_entropy)
        return info_gain

    @property
    def is_leaf(self):
        return self.info_gain == 0

    def predict(self, X):
        return np.array([self.predict_row(x) for x in X])

    def predict_row(self, x):
        if self.is_leaf:
            return self.prediction
        node = self.lhs if x[self.split_idx] >= self.split else self.rhs
        return node.predict_row(x)

    def print_node(self, level=0):
        if self.is_leaf:
            print('\t' * level, 'Leaf, predict y = %d (Gain %f)' % (self.prediction, self.info_gain))
        else:
            print('\t'*level, 'x%d >= %f (Gain %f)' % (self.split_idx, self.split, self.info_gain))
            self.lhs.print_node(level+1)
            self.rhs.print_node(level+1)










