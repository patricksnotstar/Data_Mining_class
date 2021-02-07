import numpy as np
import pandas as pd
import math
from sklearn.utils import shuffle
import copy

categorical_columns = ['workclass', 'education', 'marital-status',
                       'occupation',  'relationship', 'race', 'sex', 'native-country']

# grow(node n, dataset D, atrributes A)
#     if A is empty or then
#         n.label = majority label in D
#     else
#         for all a in A do
#             q(a) = InformationGain(a)
#         b = attribute with max q(a)
#         n.attribute = b
#         p = number of partitions for n.attribute
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
    def __init__(self, parent, attribute, data, label):
        self.parent = parent
        self.attribute = attribute
        self.data = data
        self.label = label

    def append_child(self, node):
        self.children.append(node)


def grow(node, data, attr):
    if not attr:
        node.label = node.data['income'].mode()[0]
    else:
        infoGain = {}
        for a in attr:
            if a in categorical_columns:
                infoGain[a] = calc_informationGain(
                    data, partition_cat(data, a).values())
            else:
                infoGain[a] = calc_informationGain(
                    data, partition_cont(data, a)[0])
        max_infoGain_attr = max(infoGain, key=infoGain.get)
        node.attribute = max_infoGain_attr
        if node.attribute in categorical_columns:
            num_partitions = len(partition_cat(data, node.attribute).values())
        else:
            num_partitions = len(partition_cont(data, node.attribute)[0])
        for i in range(num_partitions):
            # WIP


def calc_entropy(data):
    poor_prob = len(data[data['income'] == '>50K']) / \
        len(data) if len(data) > 0 else 0
    rich_prob = len(data[data['income'] == '<=50K']) / \
        len(data) if len(data) > 0 else 0

    if (poor_prob == 0) or (rich_prob == 0):
        return 0
    else:
        return -(poor_prob * math.log2(poor_prob) + rich_prob * math.log2(rich_prob))


def calc_informationGain(data, partitions):
    data_count = len(data)
    teeEye_entropy = 0
    for part in partitions:
        teeEye_entropy = teeEye_entropy + \
            ((len(part) / data_count) * calc_entropy(part))

    return calc_entropy(data) - (teeEye_entropy)


def partition_cat(data, column_name):
    unique_values = data[column_name].unique()
    dfDict = {teeEye: pd.DataFrame for teeEye in unique_values}
    for teeEye in dfDict.keys():
        dfDict[teeEye] = data[:][data[column_name] == teeEye]

    return dfDict


def partition_cont(data, column_name):
    partitions = []

    unique_values = data[column_name].unique()
    unique_values.sort()
    max_infoGain = 0
    split_on = 0
    for i in unique_values[::math.ceil(len(unique_values) * 0.05)]:
        temp = []
        temp.append(data[data[column_name] < i])
        temp.append(data[data[column_name] >= i])
        temp_infoGain = calc_informationGain(data, temp)
        if temp_infoGain > max_infoGain:
            partitions = copy.deepcopy(temp)
            max_infoGain = temp_infoGain
            split_on = i
    return (partitions, split_on)


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

    print(calc_informationGain(data, partition_cont(data, 'fnlwgt')))
    # print(calc_informationGain(data, partition_cat(data, 'workclass')))


if __name__ == "__main__":
    main()
