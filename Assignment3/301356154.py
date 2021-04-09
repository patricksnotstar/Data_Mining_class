import pandas as pd
import numpy as np
import csv
import copy
from efficient_apriori import apriori


def main():
    with open("BMS2.txt") as f:
        text = csv.reader(f, delimiter=" ")
        transactions = [tuple(filter(lambda x: int(x) > 0, row))
                        for row in text]
    itemsets, rules = apriori(
        transactions, min_support=0.005, min_confidence=0.7)

    allItemSets = countDict(itemsets)

    ret_closed = closedItemSets(itemsets)

    print(ret_closed)

    itemsets_copy = copy.deepcopy(itemsets)
    ret_max = maximalItemSets(itemsets_copy)

    print("All: ", allItemSets, " Closed: ", countDict(
        ret_closed), " Maximal: ", countDict(ret_max))


def countDict(itemsets):
    counter = 0
    for d in itemsets.keys():
        counter += len(itemsets[d])
    return counter


def closedItemSets(itemsets):
    suppGrps = {}
    ret = {}
    for num in itemsets.keys():
        for toop in itemsets[num].keys():
            supp = itemsets[num][toop]
            if supp in suppGrps.keys():
                if num in suppGrps[supp].keys():
                    suppGrps[supp][num].append(toop)
                else:
                    suppGrps[supp][num] = [toop]
            else:
                suppGrps[supp] = {num: [toop]}
    for supp in suppGrps.keys():
        equi_supp = suppGrps[supp]
        lengths = list(equi_supp.keys())
        while len(lengths) > 0:
            longest = max(lengths)
            for toople in equi_supp[longest]:
                if longest in ret.keys():
                    ret[longest][toople] = supp
                else:
                    ret[longest] = {}
                    ret[longest][toople] = supp
            for s in equi_supp.keys():
                for t in equi_supp[s]:
                    for toople in equi_supp[longest]:
                        if all(v in toople for v in t):
                            equi_supp[s].remove(t)
            lengths.remove(longest)
    return ret


def maximalItemSets(itemsets):
    ret = {}
    allLens = set(itemsets.keys())
    while len(allLens) > 0:
        maxLen = max(allLens)
        allLens.remove(maxLen)
        ret[maxLen] = {}
        for toop in itemsets[maxLen].keys():
            ret[maxLen][toop] = itemsets[maxLen][toop]
            for length in allLens:
                for toople in list(itemsets[length].keys()):
                    if all(v in toop for v in toople):
                        del itemsets[length][toople]
    for k in list(ret.keys()):
        if len(ret[k]) == 0:
            del ret[k]
    return ret


if __name__ == "__main__":
    main()
