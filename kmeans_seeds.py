import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

''' This method could be used against the numpy.loadtxt method'''


def construct_matrix():
    array_matrix = []
    with open('seeds_dataset.txt') as f:
        for line in f:
            nested_list = []
            row_list = line.split('\t')
            for number in row_list[:-1]:
                if number == '':
                    continue
                updated_number = float(number)
                nested_list.append(updated_number)
            array_matrix.append(nested_list)
    matrix_for_kmeans = np.array(array_matrix)
    return matrix_for_kmeans


'''The purpose of this method was to demonstrate that the plot also depicts that
ideal clusters should be 3'''


def calculate_variance():
    matrix = construct_matrix()
    matrix = PCA(n_components=2).fit_transform(matrix)

    # Determine your k range
    k_range = range(1, 14)

    # Fit the kmeans model for each n_clusters = k
    k_means_var = [KMeans(n_clusters=k).fit(matrix) for k in k_range]

    # Pull out the cluster centers for each model
    centroids = [X.cluster_centers_ for X in k_means_var]

    # Calculate the Euclidean distance from
    # each point to each cluster center
    k_euclid = [cdist(matrix, cent, 'euclidean') for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_euclid]

    # Total within-cluster sum of squares
    wcss = [sum(d ** 2) for d in dist]

    # The total sum of squares
    tss = sum(pdist(matrix) ** 2) / matrix.shape[0]

    # The between-cluster sum of squares
    bss = tss - wcss

    # elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(k_range, bss / tss * 100, 'b*-')
    ax.set_ylim((0, 100))
    plt.grid(True)
    plt.xlabel('n_clusters')
    plt.ylabel('Percentage of variance explained')
    plt.title('Variance Explained vs. k')
    plt.show()

def run_kmeans():
    matrix = construct_matrix()
    k_means = KMeans(n_clusters=3)
    k_means.fit(matrix)

    print k_means.cluster_centers_
    print k_means.labels_

'''PCA is used here for visualization purposes'''


def kmeans_with_pca():
    matrix = construct_matrix()
    dimensionalized_matrix = PCA(n_components=2).fit_transform(matrix)
    k_means = KMeans(n_clusters=3)
    k_means.fit(dimensionalized_matrix)

    x_min, x_max = dimensionalized_matrix[:, 0].min() - 5, dimensionalized_matrix[:, 0].max() - 1
    y_min, y_max = dimensionalized_matrix[:, 1].min() + 1, dimensionalized_matrix[:, 1].max() + 5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
    Z = k_means.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.figure(1)
    plt.clf()

    plt.plot(dimensionalized_matrix[:, 0], dimensionalized_matrix[:, 1], 'k.', markersize=4)
    # Plot the centroids as a white X
    centroids = k_means.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='b', zorder=8)
    plt.show()


if __name__ == '__main__':
    run_kmeans()
