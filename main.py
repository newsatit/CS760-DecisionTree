import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from DecisionTree import DecisionTree

def q3():
    (X, Y) = load_dataset('./dataset/Druns.txt')
    decision_tree = DecisionTree(list_first_candidates=True).fit(X, Y)

def q4():
    (X, Y) = load_dataset('./dataset/D3leaves.txt')
    decision_tree = DecisionTree().fit(X, Y)
    decision_tree.print_tree()

def q5_1():
    (X, Y) = load_dataset('./dataset/D1.txt')
    decision_tree = DecisionTree().fit(X, Y)
    decision_tree.print_tree()
    plot(X, Y)
    plot_decision_boundary(decision_tree, X, Y)


def q5_2():
    (X, Y) = load_dataset('./dataset/D2.txt')
    decision_tree = DecisionTree().fit(X, Y)
    decision_tree.print_tree()
    plot(X, Y)
    plot_decision_boundary(decision_tree, X, Y)

def q7():
    (X, Y) = load_dataset('./dataset/Dbig.txt')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1808)
    X_train.to_csv('extra_dataset/X_train.csv', index=None, header=True)
    Y_train.to_csv('extra_dataset/Y_train.csv', index=None, header=True)
    X_32, Y_32 = X_train.iloc[:32], Y_train.iloc[:32]
    X_32.to_csv('extra_dataset/X_32.csv', index=None, header=True)
    Y_32.to_csv('extra_dataset/Y_32.csv', index=None, header=True)
    X_128, Y_128 = X_train.iloc[:128], Y_train.iloc[:128]
    X_128.to_csv('extra_dataset/X_128.csv', index=None, header=True)
    Y_128.to_csv('extra_dataset/Y_128.csv', index=None, header=True)
    X_512, Y_512 = X_train.iloc[:512], Y_train.iloc[:512]
    X_512.to_csv('extra_dataset/X_512.csv', index=None, header=True)
    Y_512.to_csv('extra_dataset/Y_512.csv', index=None, header=True)
    X_2048, Y_2048 = X_train.iloc[:2048], Y_train.iloc[:2048]
    X_2048.to_csv('extra_dataset/X_2048.csv', index=None, header=True)
    Y_2048.to_csv('extra_dataset/Y_2048.csv', index=None, header=True)
    (count_32, error_32) = train_and_plot(X_32, Y_32, X_test, Y_test)
    (count_128, error_128) = train_and_plot(X_128, Y_128, X_test, Y_test)
    # d = {'training_size': [32, 128], 'num_node': [count_32, count_128],
    #      'test_error': [error_32, error_128]}

    (count_512, error_512) = train_and_plot(X_512, Y_512, X_test, Y_test)
    (count_2048, error_2048) = train_and_plot(X_2048, Y_2048, X_test, Y_test)
    (count_train, error_train) = train_and_plot(X_train, Y_train, X_test, Y_test)

    d = {'training_size': [32, 128, 512, 2048, 8192],
         'num_node': [count_32, count_128, count_512, count_2048, count_train],
         'test_error': [error_32, error_128, error_512, error_2048, error_train]}
    errors = pd.DataFrame(data=d)
    errors.to_csv('learning_curve.csv', index=None, header=True)

def weka():
    (X, Y) = load_dataset('./dataset/Dbig.txt')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1808)
    X_32, Y_32 = X_train.iloc[:32], Y_train.iloc[:32]
    df_32 = pd.concat([X_32, Y_32], axis=1)
    df_32.to_csv('weka_dataset/D_32.csv', index=None, header=True)
    X_128, Y_128 = X_train.iloc[:128], Y_train.iloc[:128]
    df_128 = pd.concat([X_128, Y_128], axis=1)
    df_128.to_csv('weka_dataset/D_128.csv', index=None, header=True)
    X_512, Y_512 = X_train.iloc[:512], Y_train.iloc[:512]
    df_512 = pd.concat([X_512, Y_512], axis=1)
    df_512.to_csv('weka_dataset/D_512.csv', index=None, header=True)
    X_2048, Y_2048 = X_train.iloc[:2048], Y_train.iloc[:2048]
    df_2048 = pd.concat([X_2048, Y_2048], axis=1)
    df_2048.to_csv('weka_dataset/D_2048.csv', index=None, header=True)
    df_8192 = pd.concat([X_train, Y_train], axis=1)
    df_8192.to_csv('weka_dataset/D_8192.csv', index=None, header=True)
    df_test = pd.concat([X_test, Y_test], axis=1)
    df_test.to_csv('weka_dataset/D_test.csv', index=None, header=True)

def train_and_plot(X, Y, X_test, Y_test):
    decision_tree = DecisionTree().fit(X, Y)
    # decision_tree.print_tree()
    plot_decision_boundary(decision_tree, X, Y)
    Y_predict = decision_tree.predict(X_test.to_numpy())
    test_error = 1 - metrics.accuracy_score(Y_test, Y_predict)
    num_nodes = decision_tree.count_nodes()
    print('test error : ', test_error)
    return (num_nodes, test_error)


def plot_learning_curve():
    df = pd.read_csv('weka_dataset/learning_curvel_weka.csv')
    df.plot(kind='scatter', x='training_size', y='test_error', color='red')
    plt.ylabel('test error')
    plt.xlabel('training size')
    plt.show()


def main():
   plot_learning_curve()

# def plot_decision_boundary(X, Y, clf):
#     # Parameters
#     n_classes = 2
#     plot_colors = "bry"
#     plot_step = 0.02
#
#     # Plot the decision boundary
#
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
#                          np.arange(y_min, y_max, plot_step))
#
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
#
#     plt.xlabel('x1')
#     plt.ylabel('x2')
#     plt.axis("tight")
#
#     # Plot the training points
#     for i, color in zip(range(n_classes), plot_colors):
#         idx = np.where(Y == i)
#         plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
#                     cmap=plt.cm.Paired)
#
#     plt.axis("tight")
#     plt.legend()
#     plt.show()

# Paired_r
def plot_decision_boundary(clf, X, Y, cmap='viridis'):

    fig, ax = plt.subplots()
    h = 0.02
    x_min, x_max = X.iloc[:,0].min() - 5*h, X.iloc[:,0].max() + 5*h
    y_min, y_max = X.iloc[:,1].min() - 5*h, X.iloc[:,1].max() + 5*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.25)
    ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    ax.contour(xx, yy, Z, colors='k', linewidths=0.7)

    scatter = ax.scatter(X.loc[:, 'x1'], X.loc[:, 'x2'], c=Y)

    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(), title="y")
    ax.add_artist(legend1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    # plt.scatter(X.iloc[:,0], X.iloc[:,1], c=Y, cmap=cmap, edgecolors='k')
    plt.show()

# def plot_decision_boundary(clf, X, Y, cmap='viridis'):
#     h = 0.02
#     x_min, x_max = X.iloc[:,0].min() - 10*h, X.iloc[:,0].max() + 10*h
#     y_min, y_max = X.iloc[:,1].min() - 10*h, X.iloc[:,1].max() + 10*h
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#
#     plt.figure(figsize=(5,5))
#     plt.contourf(xx, yy, Z, alpha=0.25)
#     plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
#     plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
#     plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=Y, cmap=cmap)
#     plt.legend([0, 1], title="y")
#     # plt.scatter(X.iloc[:,0], X.iloc[:,1], c=Y, cmap=cmap, edgecolors='k')
#     plt.show()

def create_dataset_2():
    X = pd.DataFrame(np.array([[0, 0], [1, 1], [0, 1], [1, 0]]), columns=['x1', 'x2'])
    Y = pd.Series(np.array([1, 1, 0, 0]), dtype='int32')
    return (X, Y)


def load_dataset(fname):
    data = np.loadtxt(fname)
    df = pd.DataFrame(data, columns=['x1', 'x2', 'y'])
    X = df.loc[:, ['x1', 'x2']]
    Y = df.loc[:, 'y'].astype('int32')
    return (X, Y)

def plot(X, Y):
    # df = pd.concat([X, Y], axis=1)
    # df.columns = ['x1', 'x2', 'y']
    # df.plot.scatter(x='x1', y='x2', c='y', colormap='viridis')

    fig, ax = plt.subplots()

    scatter = ax.scatter(X.loc[:, 'x1'], X.loc[:, 'x2'], c=Y)

    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(), title="y")
    ax.add_artist(legend1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

if __name__  == '__main__':
    main()

