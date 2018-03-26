from enum import Enum


class Distance(Enum):
    L2 = 0
    DotProduct = 1


class TensorAxis:
    N = 0
    H = 1
    W = 2
    C = 3