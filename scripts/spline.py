# Cubic spline interpolation
# adapted from https://github.com/mlab-upenn/LearningMPC/blob/master/include/LearningMPC/spline.h
import numpy as np
from scipy.interpolate import CubicSpline


# # Band matrix solver
# class BandMatrix:

#     # private members
#     __m_upper = None    # upper band
#     __m_lower = None    # lower band

#     def __init__(self, dim:int = None, n_u:int = None, n_l:int = None):
#         if dim is not None:
#             self.resize(dim, n_u, n_l)
    
#     def resize(self, dim:int, n_u:int, n_l:int):
#         assert dim > 0, "Dimension must be positive"
#         assert n_u >= 0, "Upper band must be non-negative"
#         assert n_l >= 0, "Lower band must be non-negative"
#         self.__m_upper = np.zeros((n_u + 1, dim))
#         self.__m_lower = np.zeros((n_l + 1, dim))
    
#     def dim(self):
#         if self.__m_upper is None:
#             return 0
#         return self.__m_upper.shape[1]
    
#     def __call__(self, i:int, j:int):
#         assert i >= 0, "Row index must be non-negative"
#         assert j >= 0, "Column index must be non-negative"
#         assert i < self.__m_upper.shape[0], "Row index out of bounds"
#         assert j < self.dim(), "Column index out of bounds"
#         k = j - i   # band index
#         assert k >= -self.__m_lower.shape[0] and k <= self.__m_upper.shape[0], "Band index out of bounds"
#         if k >= 0:
#             return self.__m_upper[k, i]
#         else:
#             return self.__m_lower[-k, i]


class Spline:
    def __init__(self, s: np.ndarray, x: np.ndarray):
        self.s = s
        self.x = x
        self.cs = CubicSpline(s, x)
        self.cs_d = self.cs.derivative(1)
        self.cs_dd = self.cs.derivative(2)

    def __call__(self, s):
        return self.cs(s)

    def eval_d(self, s):
        return self.cs_d(s)

    def eval_dd(self, s):
        return self.cs_dd(s)

        



