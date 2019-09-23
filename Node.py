import numpy as np
from scipy.stats import entropy


class Node:
    def __init__(self, x, y, idxs):
        self.x = x
        self.y = y
        self.idxs = idxs
        self.entropy = self.get_entropy(self.y.values[idxs])
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.val = np.argmax(np.bincount(y[idxs]))
        self.info_gain = 0
        self.find_varsplit()

    def find_varsplit(self):
        for c in range(self.col_count):
            self.find_better_split(c)
        if self.is_leaf:
            return
        x = self.split_col
        lhs = np.nonzero(x >= self.split)[0]
        rhs = np.nonzero(x < self.split)[0]
        self.lhs = Node(self.x, self.y, self.idxs[lhs])
        self.rhs = Node(self.x, self.y, self.idxs[rhs])

    def find_better_split(self, var_idx):
        x = self.x.values[self.idxs, var_idx]
        y = self.y.values[self.idxs]
        for r in range(self.row_count):
            # print('threshold: %f'%(x[r]))
            lhs = x >= x[r]
            rhs = x < x[r]

            curr_gain = self.get_info_gain(lhs, rhs)
            # print('cur infogain: %f'%curr_gain)
            if curr_gain > self.info_gain:
                self.var_idx = var_idx
                self.info_gain = curr_gain
                self.split = x[r]
        # print('x%d\'s gain is %f'%(var_idx, curr_gain))

    def get_info_gain(self, lhs, rhs):
        y = self.y.values[self.idxs]
        lhs_data = y[lhs]
        # print('lhs', y[lhs])
        # print('rhs', y[rhs])
        rhs_data = y[rhs]
        lhs_entropy = self.get_entropy(y[lhs])
        rhs_entropy = self.get_entropy(y[rhs])
        lhs_count = len(lhs_data)
        rhs_count = len(rhs_data)
        lhs_weight = lhs_count/self.row_count
        rhs_weight = rhs_count/self.row_count
        # print('entropy %f'%(self.entropy))
        # print('%f*%f + %f*%f'%(lhs_weight, lhs_entropy, rhs_weight, rhs_entropy))
        info_gain = self.entropy - (lhs_weight*lhs_entropy + rhs_weight*rhs_entropy)
        return info_gain
        # return 0
    def get_entropy(self, data):
        p_data = np.bincount(data)
        return entropy(p_data)

    @property
    def split_col(self):
        return self.x.values[self.idxs, self.var_idx]

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










