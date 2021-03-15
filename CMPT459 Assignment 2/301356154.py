import pandas as pd
import numpy as np
import queue
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def main():
    data = pd.read_csv("houshold2007.csv", na_values=['?'])

    data.dropna(inplace=True)

    data = data[data['Date'].apply(lambda x: x.endswith('/1/2007'))]

    def trim_strings(x): return x.strip() if isinstance(x, str) else x
    data = data.applymap(trim_strings)

    data = data.drop(columns=['Date', 'Time'])
    data_columns = list(data.columns)

    data_normalized = (data - data.mean())/data.std()

    data_normalized.reset_index(drop=True, inplace=True)

    data_matrix = data_normalized.to_numpy()

    minPts = 14

    plot_k_distance_diagram(data_matrix, minPts)

    labels = fit(data_normalized, 0.3, minPts)

    plot_cluster_distribution(data_normalized)


def fit(data_normalized, epsilon, minPts):
    labels = DBSCAN(data_normalized, epsilon, minPts)
    data_normalized['cluster label'] = np.full(len(data_normalized), '-1')
    for i in labels.keys():
        data_normalized.at[i, 'cluster label'] = ','.join(
            str(e) for e in labels[i])
    data_normalized.to_csv("labeled_data.csv")
    return list(data_normalized['cluster label'])


def DBSCAN(data, epsilon, minPts):
    NN = {}
    labels = {}
    lenData = len(data)
    noise = list(data.index)
    data_matrix = data.to_numpy()
    clusterId = 0
    calc_distances(data_matrix, NN, epsilon)
    for i in range(lenData):
        if i in noise:
            if len(NN[i]) >= minPts:
                if i in labels.keys():
                    labels[i].add(clusterId)
                else:
                    labels[i] = set([clusterId])
                assign_clusters(
                    data_matrix, NN[i], NN, clusterId, labels, minPts, noise, epsilon)
                clusterId += 1
    return labels


def assign_clusters(data_matrix, nn, NN, clusterId, labels, minPts, noise, epsilon):
    q = queue.Queue()
    for i in nn:
        if i not in labels.keys():
            labels[i] = set([clusterId])
            check_core(i, NN, minPts, q)
        elif clusterId not in labels[i]:
            labels[i].add(clusterId)
            check_core(i, NN, minPts, q)
        if i in noise:
            noise.remove(i)
    while not q.empty():
        for i in NN[q.get()]:
            if i not in labels.keys():
                labels[i] = set([clusterId])
                check_core(i, NN, minPts, q)
            elif clusterId not in labels[i]:
                labels[i].add(clusterId)
                check_core(i, NN, minPts, q)
            if i in noise:
                noise.remove(i)


def check_core(idx, NN, minPts, q):
    if len(NN[idx]) >= minPts:
        q.put(idx)


def calc_distances(data, NN, epsilon):
    for i in range(len(data)):
        distances = np.linalg.norm(data[i] - data[i:, None], axis=-1)
        k = i
        for j in range(len(distances)):
            if distances[j] <= epsilon:
                if i in NN.keys():
                    NN[i].add(k)
                else:
                    NN[i] = set([k])
                if k in NN.keys():
                    NN[k].add(i)
                else:
                    NN[k] = set([i])
            k += 1


def plot_k_distance_diagram(data, minPts):
    neighbours = NearestNeighbors(n_neighbors=(minPts)).fit(data)
    distances, _ = neighbours.kneighbors(data)

    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    fig = plt.figure()
    plt.plot(distances)

    plt.xlabel("Points")
    plt.ylabel("Epsilon")
    plt.title("K Distance Diagram for K=" + str(minPts))

    plt.savefig("k-distance-diagram.png")


def plot_cluster_distribution(data):
    grouped = data.groupby("cluster label")["cluster label"].count()
    print(grouped)

    fig = plt.figure()
    grouped.plot(kind="bar")

    plt.xlabel("Cluster labels")
    plt.ylabel("Number of Data Points")
    plt.title("Label Frequency")

    plt.savefig("cluster_distribution.png")


if __name__ == "__main__":
    main()
