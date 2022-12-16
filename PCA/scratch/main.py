from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
from sklearn.decomposition import PCA

"""
https://www.linkedin.com/pulse/pca-second-principal-component-have-orthogonal-first-mukesh-manral/
"""


if __name__ == "__main__":
    A = array([
        [1, 2],
        [3, 4],
        [5, 6]
    ])
    print(A)
    M = mean(A, axis=0)
    print(mean(A, axis=0))
    C = A - M
    print(C)
    V = cov(C.T)
    print(V)
    values, vectors = eig(V)
    print(values)
    print(vectors)
    P = vectors.T.dot(C.T)
    print(P.T)

    print("=" * 50)
    pca = PCA(2)
    pca.fit(A)
    print(pca.components_)
    print(pca.explained_variance_)
    B = pca.transform(A)
    print(B)
