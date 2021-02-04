import numpy as np
from sympy import *
from sympy.abc import x, y
import matplotlib.pyplot as plt
import scipy

def get_m0(n: int):
    """
    get a fundamental matrix in statistics, "centering matrix" that is used to transform data to
    deviations from their mean
    :param n: length of data
    :return: $M_0$
    """
    return (-1 / n) * np.ones((n, n)) + np.eye(n)


def sum(array):
    n = array.shape[0]
    i = np.ones(n)

    return np.dot(i, array)

def mean(array):
    import numpy as np
    n = array.shape[0]
    i = np.ones(n)
    return (1 / n) * np.dot(i, array)


def mean_deviation(array):
    n = array.shape[0]
    m0 = get_m0(n)
    return np.matmul(m0, array)


def sum_of_squares(array):
    return np.sum(array ** 2)


def sum_of_squared_deviation(array):
    return sum_of_squares(mean_deviation(array))


def sum_of_squres_matrix(a, b):
    z = np.c_[a, b]
    return z.T.matmul(get_m0(z.shape[0])).matmul(z)


def rref(matrix):
    m = Matrix(matrix)
    return np.array(m.rref()[0].tolist()).astype(np.float64)

def rank(matrix):
    return np.linalg.matrix_ranK(matrix)

def det(matrix):
    return np.linalg.det(matrix)

def inv(matrix):
    return np.linalg.inv(matrix)

def l1_norm(matrix):
    return np.abs(np.array(matrix)).sum()

def l2_norm(matrix):
    return np.sqrt((np.array(matrix) ** 2).sum())

def dot(arr1, arr2):
    return np.dot(np.array(arr1), np.array(arr2))

def arccos(angle):
    return np.arccos(angle)

def arccos_deg(angle):
    return arccos(angle) * 180 / np.pi

def orthogonal(arr1, arr2):
    return dot(arr1, arr2) == 0

def normalize(v):
    return v/l2_norm(v)

def inner_product(x, y, A):
    x_arr = np.array(x)
    y_arr = np.array(y)
    A_arr = np.array(A)
    return np.matmul(np.matmul(x_arr, A_arr), y_arr.T)
def proj_matrix(U):
    U_arr = np.array(U)

    return U_arr.dot(inv(U_arr.T.dot(U_arr))).dot(U_arr.T)

def proj(v, U):
    v_arr = np.array(v)

    return proj_matrix(U).dot(v_arr)

def gram_schmidt(B):
    B_arr = np.array(B)
    for i in range(B_arr.shape[1]):
        if i == 0:
            U = normalize(B_arr[:, i])
            U = U.reshape(U.shape[0], -1)
        else:
            U_j = normalize(B_arr[:, i] - proj(B_arr[:, i], U))
            U = np.c_[U, U_j]
    return U

def rotate(x, deg):
    x_arr = np.array(x)
    rad = deg*np.pi/180
    return np.matmul(np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]]), x_arr)

def det_laplace_expansion(A, checked=False):
    arr = np.array(A)
    if not checked:
        if len(arr.shape) == 1:
            assert arr.shape[0] == 1, "Input matrix is not square."
        else:
            assert arr.shape[0] == arr.shape[1], "Input matrix is not square."
    check = True
    if (arr.shape[0] == 1):
        return arr
    else:
        det = 0
        for i in range(arr.shape[0]):
            row_index = np.array([x for x in range(arr.shape[0]) if x != i])[:,np.newaxis]
            print(arr.shape[0])
            print(row_index)
            column_index = np.array([x for x in range(arr.shape[1]) if x != 0])
            print(column_index)
            print(arr[row_index, column_index])
            det += ((-1)**i) * det_laplace_expansion(arr[row_index, column_index], checked)
        return det

def trace(A):
    arr = np.array(A)
    return np.trace(A)

def charpoly(A):
    M = Matrix(A)
    return M.charpoly(x).as_expr()

def eig(A):
    M = np.array(A)
    return np.linalg.eig(M)

def cholesky(lA):
    arrA = np.array(lA)
    return np.linalg.cholesky(arrA)


def plotVectors(vecs, cols, alpha=1):
    """
    Plot set of vectors.

    Parameters
    ----------
    vecs : array-like
        Coordinates of the vectors to plot. Each vectors is in an array. For
        instance: [[1, 3], [2, 2]] can be used to plot 2 vectors.
    cols : array-like
        Colors of the vectors. For instance: ['red', 'blue'] will display the
        first vector in red and the second in blue.
    alpha : float
        Opacity of vectors

    Returns:

    fig : instance of matplotlib.figure.Figure
        The figure of the vectors
    """
    plt.figure()
    plt.axvline(x=0, color='#A9A9A9', zorder=0)
    plt.axhline(y=0, color='#A9A9A9', zorder=0)

    for i in range(len(vecs)):
        x = np.concatenate([[0,0],vecs[i]])
        plt.quiver([x[0]],
                   [x[1]],
                   [x[2]],
                   [x[3]],
                   angles='xy', scale_units='xy', scale=1, color=cols[i],
                   alpha=alpha)

def eigen_decomposition(lA):
    arrA = np.array(lA)
    results = eig(arrA)
    assert det(results[1]) != 0, "Defective matrix cannot be diagonalized."
    return (results[1], np.diag(results[0]), inv(results[1]))

def svd(lA):
    arrA = np.array(lA)
    U, S, VT = scipy.linalg.svd(arrA)
    Sigma = np.zeros(arrA.shape)
    k = min(arrA.shape)
    Sigma[:k, :k] = np.diag(S)
    return U, Sigma, VT

def pinv(lA):
    arrA = np.array(A)
    U, s, VT = np.linalg.svd(A)
    d = 1.0/s
    D = np.zeros(A.shape)
    k = min(A.shape)
    D[:k, :k] = np.diag(d)
    return (VT.T).dot((D.T).dot(U.T))

def matrix_approximate(lA, n_components):
    arrA = np.array(lA)
    U, Sigma, VT = svd(arrA)
    Sigmak = Sigma[:, :n_components]
    VTk = VT[:n_components, :]
    print(U.shape, Sigmak.shape, VTk.shape)
    return U.dot(Sigmak.dot(VTk))

class LinearRegression:

    def __init__(self):
        self._params_ = None
        self._intercept_ = None
        self._coef_ = None

    @property
    def params_(self):
        return self._params_

    @params_.setter
    def params_(self, params):
        import numpy as np
        self._params_ = np.array(params)

    @property
    def intercept_(self):
        return self._intercept_

    @intercept_.setter
    def intercept_(self, intercept):
        import numpy as np
        self._intercept_ = np.array(intercept)

    @property
    def coef_(self):
        return self._coef_

    @coef_.setter
    def coef_(self, coef):
        import numpy as np
        self._coef_ = np.array(coef)

    def fit(self, X, y):
        n = X.shape[0]
        ones = np.ones(n)
        feat = np.c_[X, ones]
        inv = np.linalg.inv(np.matmul(feat.T, feat))
        self.params_ = np.matmul(np.matmul(inv, feat.T), y)
        self.coef_ = self.params_[0]
        self.intercept_ = self.params_[1]
        print(self.params_)

    def predict(self, x):
        n = x.shape[0]
        feat = np.c_[x, np.ones(n)]
        return np.matmul(feat, self.params_)