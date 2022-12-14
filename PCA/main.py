import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

"""
Debug PCA: https://www.askpython.com/python/examples/principal-component-analysis
"""

def PCA(X, num_components):
    X_meaned = X - np.mean(X, axis=0)
    print(X_meaned)
    cov_mat = np.cov(X_meaned, rowvar=False)
    print(cov_mat)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    print(eigen_values)
    print("=" * 50)
    print(eigen_vectors)
    print("=" * 50)
    sorted_index = np.argsort(eigen_values)[::-1]
    print(sorted_index)
    print("=" * 50)
    sorted_eigenvalue = eigen_values[sorted_index]
    print(sorted_eigenvalue)
    print("=" * 50)
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    print(sorted_eigenvectors)
    print("=" * 50)
    n_components = 2
    eigenvectors_subset = sorted_eigenvectors[:, 0:n_components]
    print(eigenvectors_subset)
    print("=" * 50)
    X_reduced = np.dot(eigenvectors_subset.transpose(), X_meaned.transpose()).transpose()
    print(X_reduced)
    return X_reduced


if __name__ == "__main__":
    X = np.random.randint(10, 50, 100).reshape(20, 5)
    print(X)
    print(np.mean(X, axis=0))

    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    # data = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])
    #
    # x = data.iloc[:, 0:4]
    #
    # target = data.iloc[:, 4]

    mat_reduced = PCA(X, 2)
    print(mat_reduced.shape)
    # principal_df = pd.DataFrame(mat_reduced, columns=['PCA1', 'PCA2'])
    # principal_df = pd.concat([principal_df, pd.DataFrame(target)], axis=1)
    # plt.figure(figsize=(6, 6))
    # sb.scatterplot(data=principal_df, x='PCA1', y='PCA2', hue='target', s=60, palette='icefire')
    # plt.show()
