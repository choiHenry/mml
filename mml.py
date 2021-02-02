import numpy as np
from sympy import *


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

# def gram_schmidt(arr):
#     arr1 = np.array(arr)
#     for i in range(arr1.shape[1]):
#         if (i == 0):
#             u = normalize(arr1[:, i])
#         else:
#             v = arr1[:, i]
#             s = v.copy()
#             for j in range(i):
#                 if len(u.shape) == 1:
#                     v = v - proj(s, u)
#                 else:
#                     v = v - proj(s, u[:, j])
#             v = normalize(v)
#             u = np.c_[u, v]
#     return u

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
            U = B_arr[:, i]
            U = U.reshape(U.shape[0], -1)
        else:
            U_j = normalize(B_arr[:, i] - proj(B_arr[:, i], U))
            U = np.c_[U, U_j]
    return U



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