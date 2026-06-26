import numpy as np


def poly44(coeff, x, y):
    """
    四阶多项式计算函数 (支持标量或矩阵输入)
    """
    x, y = np.asanyarray(x), np.asanyarray(y)

    res = coeff[0] + \
          coeff[1] * x + coeff[2] * y + \
          coeff[3] * (x ** 2) + coeff[4] * x * y + coeff[5] * (y ** 2) + \
          coeff[6] * (x ** 3) + coeff[7] * (x ** 2) * y + coeff[8] * x * (y ** 2) + coeff[9] * (y ** 3) + \
          coeff[10] * (x ** 4) + coeff[11] * (x ** 3) * y + coeff[12] * (x ** 2) * (y ** 2) + coeff[13] * x * (
          y ** 3) + coeff[14] * (y ** 4)
    return res


def differentiate_poly44(coeff, x, y):
    """
    计算多项式在 (x, y) 处的空间偏导数 (梯度)
    返回: (fx, fy) 分别对应 x 方向和 y 方向的偏导数
    """
    x, y = np.asanyarray(x), np.asanyarray(y)
    n = len(coeff)
    fx = coeff[1] + 2 * coeff[3] * x + coeff[4] * y + \
         3 * coeff[6] * (x ** 2) + 2 * coeff[7] * x * y + coeff[8] * (y ** 2) + \
         4 * coeff[10] * (x ** 3) + 3 * coeff[11] * (x ** 2) * y + 2 * coeff[12] * x * (y ** 2) + coeff[13] * (
         y ** 3)


    fy = coeff[2] + coeff[4] * x + 2 * coeff[5] * y + \
         coeff[7] * (x ** 2) + 2 * coeff[8] * x * y + 3 * coeff[9] * (y ** 2) + \
         coeff[11] * (x ** 3) + 2 * coeff[12] * (x ** 2) * y + 3 * coeff[13] * x * (y ** 2) + 4 * coeff[14] * (
         y ** 3)

    return fx, fy