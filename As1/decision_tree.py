import numpy as np
from numpy.lib.arraysetops import unique
from numpy.lib.shape_base import split
import pandas as pd
import math
import copy
import reverse_geocoder as rg

categorical_columns = ['workclass', 'education', 'marital-status',
                       'occupation',  'relationship', 'race', 'sex', 'native-country']

used_cont_values = {}


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
        node.label = data['income'].mode()[0]
        for key in partitions.keys():
            if best_splitAttr in categorical_columns:
                child = Node(node, key, None, None)
            else:
                child = Node(node, best_splitValue, None, key)
            attr_copy = copy.deepcopy(attr)
            if key in categorical_columns:
                attr_copy.remove(best_splitAttr)
            node.append_child(child)
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


def prune(root, curr_node, val_data):
    if len(curr_node.children) == 0:
        og_accuracy = calc_accuracy(root, val_data)
        curr_node.parent.children.remove(curr_node)
        new_accuracy = calc_accuracy(root, val_data)
        if og_accuracy > new_accuracy:
            curr_node.parent.children.append(curr_node)
            return
    else:
        for child in curr_node.children:
            prune(root, child, val_data)
        if len(curr_node.children) == 0:
            og_accuracy = calc_accuracy(root, val_data)
            curr_node.parent.children.remove(curr_node)
            new_accuracy = calc_accuracy(root, val_data)
            if og_accuracy > new_accuracy:
                curr_node.parent.children.append(curr_node)
                return


def calc_accuracy(root, data):
    temp = data.apply(traverse_tree, node=root, axis=1)
    data['predicted_label'] = temp
    return len(data[data['income'] == data['predicted_label']])*100/len(data)


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
#
#
# def calc_entropy(data):
#     poor_prob = len(data[data['income'] == '>50K']) / \
#         len(data) if len(data) > 0 else 0
#     rich_prob = len(data[data['income'] == '<=50K']) / \
#         len(data) if len(data) > 0 else 0
#
#     if (poor_prob == 0) or (rich_prob == 0):
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
    for val in used_cont_values[column_name]:
        # Drop the unique values that have been used to split the same attribute
        np.delete(unique_values, np.where(unique_values == val), 0)
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
            used_cont_values[column_name].append(i)
    return partitions, split_on


def main():

    # Initializing
    data = pd.read_csv('data/adult.data.csv')
    test_data = pd.read_csv('data/adult.test.csv')

    categorical_columns = ['workclass', 'education', 'marital-status',
                           'occupation',  'relationship', 'race', 'sex', 'native-country']

    categories = list(data.columns)
    categories.remove('income')
    for attr in categories:
        if attr not in categorical_columns:
            used_cont_values[attr] = []

    data['predicted-label'] = np.nan

    # Data Cleaning

    data = data[data['income'] != '?']

    def trim_strings(x): return x.strip() if isinstance(x, str) else x
    data = data.applymap(trim_strings)
    test_data = test_data.applymap(trim_strings)

    rich = data[data['income'] == '>50K']
    poor = data[data['income'] == '<=50K']

    for attr in categorical_columns:
        richMode = rich[attr].mode()[0].strip()
        poorMode = poor[attr].mode()[0].strip()

        data.loc[((data[attr] == "?") & (data['income'] == '>50K')),
                 attr] = richMode
        data.loc[((data[attr] == "?") & (data['income'] == '<=50K')),
                 attr] = poorMode

        test_data.loc[(test_data[attr] == "?"),
                      attr] = test_data[attr].mode()[0].strip()

    # train-test-split
    split_size = math.floor(len(data) / 5)
    data_chunks = []
    start = 0
    stop = split_size
    for _ in range(5):
        data_chunks.append(data[start:stop])
        start = start + split_size
        stop = stop + split_size

    accuracies = []

    for i in range(5):
        evaluation_data = data_chunks[i]
        training_data = pd.concat(data_chunks[0: i] + data_chunks[i+1:])
        validation_data = training_data[0: math.floor(
            len(training_data) * 0.1)]
        training_data = training_data[len(validation_data):]
        root = Node(None, None, None, None)
        grow(root, training_data, categories, 0.8 + i*0.05)
        prune(root, root, validation_data)
        accuracies.append(calc_accuracy(root, evaluation_data))

    avg_acc = sum(accuracies) / len(accuracies)
    print(accuracies)
    print("Average accuracy: ", avg_acc)
    best_param = accuracies.index(max(accuracies))
    root = Node(None, None, None, None)
    training_data = data[0: math.floor(len(data) * 0.1)]
    validation_data = data[len(training_data):]
    grow(root, training_data, categories, accuracies[best_param])
    prune(root, root, validation_data)
    temp = test_data.apply(traverse_tree, node=root, axis=1)
    test_data['predicted-label'] = temp
    test_data.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()
