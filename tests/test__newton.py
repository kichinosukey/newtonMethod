import numpy as np

def dfx(X):
    return np.array([[4*(X[0][0]-2)**3+2*(X[0][0]-2*X[1][0])], [-4*(X[0][0] - 2*X[1][0])]])*(-1)

def ddfx(X):
    return np.array([
        [12*(X[0][0] - 2)**2 + 2, -4],
        [-4, 8]
    ])

def test__newton():

    X0 = np.array([[0], [3]])
    iteration = 30
    X_new = X0
    for i in range(iteration):
        diff = np.dot(np.linalg.inv(ddfx(X_new)), dfx(X_new))
        X_new = X_new + diff
    np.testing.assert_allclose(X_new, np.array([[2], [1]]), rtol=1e-05)