import numpy as np
import pandas as pd
import math
from sklearn.utils import shuffle


# grow(node n, dataset D, atrributes A)
#     if A is empty or then
#         n.label = majority label in D
#     else
#         for all a in A do
#             q(a) = InformationGain(a)
#         b = attribute with max q(a)
#         n.attribute = b
#         p = number of partitoins for n.attribute
#         for i from 1 to p do
#             Di = { d in D | d belongs to partion i of b}
#             create new node sa child of n
#             grow(ni, Di, A - {b})

# prune(dataset D, node n)
#     original_score = test(D, n)
#     if n has no children
#         return
#     for each child of n
#         prune(D)


class Node:
    def __init__(self, parent, attribute, attribute_value, label):
        self.parent = parent
        self.attribute = attribute
        self.attribute_value = attribute_value
        self.label = label

    def append_child(self, node):
        self.children.append(node)


# def grow(Node n):


def main():

    data = pd.read_csv('adult.data.csv')
    test_data = pd.read_csv('adult.test.csv')

    categorical_columns = ['workclass', 'education', 'marital-status',
                           'occupation',  'relationship', 'race', 'sex', 'native-country']

    # Data Cleaning

    data = data[data['income'] != '?']

    def trim_strings(x): return x.strip() if isinstance(x, str) else x
    data = data.applymap(trim_strings)

    rich = data[data['income'] == '>50K']
    poor = data[data['income'] == '<=50K']

    for attr in categorical_columns:
        richMode = rich[attr].mode()[0].strip()
        poorMode = poor[attr].mode()[0].strip()

        data.loc[((data[attr] == "?") & (data['income'] == '>50K')),
                 attr] = richMode
        data.loc[((data[attr] == "?") & (data['income'] == '<=50K')),
                 attr] = poorMode

    # train-test-split

    total_numRows = data['income'].count()
    validation_numRows = math.floor((total_numRows * 0.1))

    validation_data = data[0: validation_numRows]

    trainingRange = math.floor((total_numRows - validation_numRows) / 5)

    training_sets = []

    for i in range(5):
        start = validation_numRows + trainingRange * i
        stop = validation_numRows + trainingRange * (i+1)
        training_sets.append(
            data[start: stop])


if __name__ == "__main__":
    main()
