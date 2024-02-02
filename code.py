import numpy as np
import matplotlib.pyplot as plt


def aproksymacja_MNK(x, y, n):
    A, At = a_matrix(x, y, n)
    M = np.dot(At, A)
    M1 = np.array(M)
    L, Lt = l_matrix(M1)
    v = l_lower(L=L, At=At, y=y)
    a_v = l_upper(Lt=Lt, v=v)
    wielomian = np.poly1d(a_v[::-1])
    print("Współczynniki wielomianu aproksymacji: ")
    print(a_v)
    x_reg = np.linspace(start=np.min(x), stop=np.max(x), num=1000)
    y_reg = wielomian(x_reg)
    plt.scatter(x, y, label="punkty(x,y)")
    plt.plot(x_reg, y_reg, "r", label=f'Wilomian Aproksymacji stopnia {n}')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Aproksymacja MNK")
    plt.legend()
    plt.show()


def a_matrix(x, y, n):
    one = [1]*len(x)
    A = np.matrix(data=[one]*(n+1)).transpose()
    for i in range(0, n+1):
        for j in range(0, len(x)):
            A[j, i] = x[j]**i
    At = A.transpose()
    return A, At


def l_matrix(A):
    L = np.zeros_like(A, dtype=float)
    for i in range(len(A)):
        for j in range(i + 1):
            if i == j:
                L[i, j] = np.sqrt(A[i, j] - np.dot(L[i, :j], L[i, :j]))
            else:
                L[i, j] = (A[i, j] - np.dot(L[j, :j], L[i, :j]))/L[j, j]
    return L, L.transpose()


def l_lower(L, At, y):
    v = np.zeros(shape=L.shape[0])
    b = np.matmul(At, y)
    b = b.transpose()
    for i in range(0, len(v)):
        v[i] = (b[i] - np.dot(v[:i], L[i, :i]))/L[i, i]
    return v


def l_upper(Lt, v):
    a = np.zeros(shape=Lt.shape[0])
    b = v
    for i in range(len(v)-1, -1, -1):
        a[i] = (b[i] - np.sum(a[i+1:] * Lt[i, i+1:]))/Lt[i, i]
    return a


if __name__ == "__main__":
    x = np.array([-10., -5.,  0.,  5., 10.])
    y = np.array([-200.,  137.5,  0., -237.5, -200.])
    aproksymacja_MNK(x, y, 3)
