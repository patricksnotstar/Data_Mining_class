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
    def __init__(self, parent, split_value, label, lessThan):
        self.parent = parent
        self.attribute = None
        self.split_value = split_value
        self.label = label
        self.lessThan = lessThan
        self.children = []

    def append_child(self, node):
        self.children.append(node)


def grow(node, data, attr, majority):
    if len(data) == 0:
        return
    richMajor = len(data[data['income'] == '>50K']) / len(data) >= majority
    poorMajor = len(data[data['income'] == '<=50K']) / len(data) >= majority
    if len(attr) == 0 or (richMajor or poorMajor):
        node.label = data['income'].mode()[0]
        return
    else:
        best_splitAttr, best_splitValue, partitions = calc_bestPartition(
            data, attr)
        node.attribute = best_splitAttr
        for key in partitions.keys():
            if best_splitAttr in categorical_columns:
                child = Node(node, key, "Not Leaf", None)
            else:
                child = Node(node, best_splitValue, 'Not Leaf', key)
            attr_copy = copy.deepcopy(attr)
            attr_copy.remove(best_splitAttr)
            node.append_child(child)
            if len(partitions[key]) == 0:
                node.label = data['income'].mode()[0]
                return
            grow(child, partitions[key], attr_copy, majority)


def traverse_tree(data, node):
    if len(node.children) > 0:
        if node.attribute in categorical_columns:
            for child in node.children:
                if child.split_value == data[node.attribute]:
                    temp = traverse_tree(data, child)
                    return temp

        else:
            if node.children[0].lessThan and data[node.attribute] <= node.children[0].split_value:
                temp = traverse_tree(
                    data, node.children[0])
                return temp
            elif len(node.children) > 1:
                temp = traverse_tree(
                    data, node.children[1])
                return temp
    else:
        return node.label


def calc_bestPartition(data, attr):
    best_infoGain = -1
    best_splitAttr = None
    best_splitValue = None
    partitions = None
    for a in attr:
        if a in categorical_columns:
            partitions_dict = partition_cat(data, a)
            infoGain = calc_informationGain(data, partitions_dict)
            if infoGain > best_infoGain:
                best_infoGain = infoGain
                best_splitAttr = a
                best_splitValue = partitions_dict.keys()
                partitions = partitions_dict
        else:
            temp_partitions, split_on = partition_cont(data, a)
            infoGain = calc_informationGain(data, temp_partitions)
            if infoGain > best_infoGain:
                best_infoGain = infoGain
                best_splitAttr = a
                best_splitValue = split_on
                partitions = temp_partitions
    return best_splitAttr, best_splitValue, partitions


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
    for key in partitions.keys():
        teeEye_entropy = teeEye_entropy + \
            ((len(partitions[key]) / data_count)
             * calc_entropy(partitions[key]))

    return calc_entropy(data) - (teeEye_entropy)


def partition_cat(data, column_name):
    unique_values = data[column_name].unique()
    dfDict = {teeEye: pd.DataFrame for teeEye in unique_values}
    for teeEye in dfDict.keys():
        dfDict[teeEye] = data[:][data[column_name] == teeEye]
    return dfDict


def partition_cont(data, column_name):
    partitions = {}
    unique_values = data[column_name].unique()
    unique_values.sort()
    max_infoGain = 0
    split_on = 0
    for i in unique_values[::math.ceil(len(unique_values) * 0.05)]:
        temp = {}
        temp[True] = data[data[column_name] <= i]
        temp[False] = data[data[column_name] > i]
        temp_infoGain = calc_informationGain(data, temp)
        if temp_infoGain > max_infoGain:
            partitions = copy.deepcopy(temp)
            max_infoGain = temp_infoGain
            split_on = i
    return partitions, split_on


def main():

    data = pd.read_csv('adult.data.csv')
    test_data = pd.read_csv('adult.test.csv')

    categorical_columns = ['workclass', 'education', 'marital-status',
                           'occupation',  'relationship', 'race', 'sex', 'native-country']

    categories = list(data.columns)
    categories.remove('income')

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

    root = Node(None, None, "Not Leaf", None)
    grow(root, data, categories, 0.80)

    temp = data.apply(traverse_tree, node=root, axis=1)
    data['predicted_label'] = temp
    print(data.head(20))
    print(len(data[data['income'] == data['predicted_label']])*100/len(data))


if __name__ == "__main__":
    main()
