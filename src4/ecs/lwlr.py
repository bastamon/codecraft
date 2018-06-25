
from math import *

# def det(Samplef):
#     X = Samplef.copy()
#     detX=[]
#     if len(X[0]) == 1:
#         for dex, i in enumerate(X):
#             detX.append(reduce(lambda x, y: x - y, i))
#     else:
#         mult = []
#         for dex ,i in enumerate(X):
#             mult.append(reduce(lambda x, y: x * y, i))
#             if dex != 0:
#                 detX.append(mult[dex] - mult[dex-1])
#     return detX


def det(m):
    if len(m) <= 0:
        return None
    elif len(m) == 1:
        return m[0][0]
    else:
        s = 0
        for i in range(len(m)):
            n = [[row[a] for a in range(len(m)) if a != i] for row in m[1:]]
            if i % 2 == 0:
                s += m[0][i] * det(n)
            else:
                s -= m[0][i] * det(n)
        return s


def addwmatrix(x, X, Y, k = 0.1):
    m = X.shapesize[0]
    W = [[0 for _ in len(m)]for _ in len(m)]
    for i in range(m):
        xi = X[i][0]
        x.append(x)
        W[i,i] = exp(sqrt(pow(x-xi,2))/(-2*k**2))
    xWx = X.T * W * X
    if det(xWx) == 0:
        print 'xWx is a singular matrix'
        return
    w = xWx.I*X.T*W*Y
    return w




def wmatrix(x, y):
    if type(x) not in (int, float, complex) or type(y) not in (int, float, complex):
        print 'data type error'
        return
    w = [[] for i in range(x)]
    for i in range(x):
        for j in range(y):
            w[i].append(random.random()/x)
            # print w[i]
    return w
