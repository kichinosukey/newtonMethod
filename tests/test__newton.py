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

        rtol = 1e-05
        if i == 0:
            np.testing.assert_allclose(X_new, np.array([[0.666667], [0.333333]]), rtol=rtol)
        elif i == 1:
            np.testing.assert_allclose(X_new, np.array([[1.111111], [0.555556]]), rtol=rtol)
        elif i == 2:
            np.testing.assert_allclose(X_new, np.array([[1.407407], [0.703704]]), rtol=rtol)
        elif i == 3:
            np.testing.assert_allclose(X_new, np.array([[1.604938], [0.802469]]), rtol=rtol)

    np.testing.assert_allclose(X_new, np.array([[2], [1]]), rtol=1e-05)