from enum import Enum


class SortMethod(Enum):
    ORACLE = 0 # uses an oracle order provided externally
    PPR_T2 = 1
    EGO_NET_1 = 2
    DEGREE = 3
