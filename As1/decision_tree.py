import numpy as np
import pandas as pd
import math


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

    pd.options.mode.chained_assignment = None  # default='warn'

    data = pd.read_csv('adult.data.csv')
    test_data = pd.read_csv('adult.test.csv')

    categorical_columns = ['workclass', 'education', 'marital-status',
                           'occupation',  'relationship', 'race', 'sex', 'native-country']

    data = data[data['income'] != '?']

    # data = data[data['workclass'].str.replace(max(data.groupby(['income']).count()))]

    rich = data[data['income'] == '>50K']
    poor = data[data['income'] == '<=50K']

    for attr in categorical_columns:
        richMode = rich[attr].mode()[0].strip()
        poorMode = poor[attr].mode()[0].strip()

        rich[attr] = rich[attr].replace(
            to_replace=r'\?', value=richMode, regex=True)
        poor[attr] = poor[attr].replace(
            to_replace=r'\?', value=poorMode, regex=True)

    newDf = pd.concat([rich, poor], axis=0)
    # data['workclass'] = data[data['income'] == '>50K']['workclass'].replace(
    #     to_replace=r'\?', value=rich['workclass'].mode()[0], regex=True)
    # data['workclass'] = data[data['income'] == '<=50K']['workclass'].replace(
    #     to_replace=r'\?', value=rich['workclass'].mode()[0], regex=True)

    # data['workclass'] = np.where(
    # data['workclass'] == '?' and data['income'] == '>50K', richMode, data["workclass"])

    # print(data['workclass'].value_counts())
    # print(data[data['income'] == '>50K']['workclass'])
    print(newDf['workclass'].value_counts())


if __name__ == "__main__":
    main()
