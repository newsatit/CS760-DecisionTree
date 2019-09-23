import pandas as pd
import numpy as np
from DecisionTree import DecisionTree

def main():
    # f = open("./dataset/D1.txt", "r")
    data = np.loadtxt("./dataset/D2.txt")
    df = pd.DataFrame(data, columns=['x1', 'x2', 'y'])
    x = df.loc[:, ['x1', 'x2']]
    y = df.loc[:, 'y'].astype('int32')

    dtree = DecisionTree().fit(x, y)
    dtree.print_tree()

if __name__  == '__main__':
    main()

