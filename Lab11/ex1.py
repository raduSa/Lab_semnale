import numpy as np
import matplotlib.pyplot as plt

trend_func = lambda x: x**2 + 5*x + 10
seasonal_func = lambda x: 5*np.sin(10*np.pi*x) + 3*np.cos(40*np.pi*x)
result_func = lambda x, y, z: x + y + z

N = 1000
x = np.linspace(0, 10, N)
trend = np.vectorize(trend_func)(x)
seasonal = np.vectorize(seasonal_func)(x)
noise = np.random.normal(0, 5, size=1000)
series = np.vectorize(result_func)(trend, seasonal, noise)

L = 20

K = N - L + 1
X = np.column_stack([series[i:i+L] for i in range(K)])

print(X.shape)

XXT = np.dot(X, X.T)
XTX = np.dot(X.T, X)

eigs_XXT = np.linalg.eigvals(XXT)
eigs_XTX = np.linalg.eigvals(XTX)

# print(eigs_XXT) # -> 10 eigs
# print(eigs_XTX) # -> 991 eigs

U, singular_X, Vh = np.linalg.svd(X, full_matrices=False)
# print(eigs_XXT, singular_X ** 2) # -> sigma_i = sqrt(lambda_i)
print(U.shape, singular_X.shape, Vh.shape)

def hankelize(X):    
    series_component = np.zeros(N)
    counts = np.zeros(N)

    for i in range(L):
        for j in range(K):
            series_component[i + j] += X[i, j]
            counts[i + j] += 1

    return series_component / counts

x_hat = list()
for i in range(L):
    x_hat_i = hankelize(np.outer(U[:, i], Vh[i, :]) * singular_X[i])
    x_hat.append(x_hat_i)

# print(series[:10])
# print(np.sum(np.array(x_hat), axis=0)[:10])
print(np.allclose(np.array(series), np.sum(np.array(x_hat), axis=0)))

# plot incremental sum of components

num_plots = 4
fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3*num_plots), sharex=True)

sum = np.zeros_like(series)

for idx, k in enumerate(range(0, L, L//4)):
    sum += np.sum(x_hat[k : k + L//4], axis=0)

    axes[idx].plot(series, label='Original', color='black')
    axes[idx].plot(sum, label=f'Components 1–{min(k+L//4, L)}')    
    axes[idx].legend()

plt.suptitle('SSA Reconstruction')
plt.savefig(fname=f'Lab11/ex1_fig1.pdf')
plt.show()