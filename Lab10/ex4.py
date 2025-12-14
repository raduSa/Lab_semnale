import numpy as np

def get_roots(coefs):
    n = len(coefs)
    
    companion = np.zeros((n, n))
    companion[:, -1] = -coefs
    companion[1 :, : -1] = np.eye(n - 1)

    eigenvals = np.linalg.eigvals(companion)

    return eigenvals

def is_stationary(x_star):
    polynomial = np.append(np.array([1]), x_star)
    if polynomial[-1] != 0:
        polynomial = polynomial / polynomial[-1]

    # print(polynomial)

    roots = get_roots(polynomial)
    return np.all(np.abs(roots) < 1)

# print(get_roots(np.array([1, 2, 3])))
# print(is_stationary(np.array([1, 2, 3])))