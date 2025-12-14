import numpy as np
import matplotlib.pyplot as plt
from l1regls import l1regls
from cvxopt import normal, matrix
from ex4 import is_stationary
from statsmodels.tsa.ar_model import AutoReg

trend_func = lambda x: x**2 + 5*x + 10
seasonal_func = lambda x: 5*np.sin(10*np.pi*x) + 3*np.cos(40*np.pi*x)
result_func = lambda x, y, z: x + y + z

N = 1000
x = np.linspace(0, 10, N)
trend = np.vectorize(trend_func)(x)
seasonal = np.vectorize(seasonal_func)(x)
noise = np.random.normal(0, 5, size=1000)
series = np.vectorize(result_func)(trend, seasonal, noise)

p = 100
nonzero_vals = 30
Y = np.zeros((N-p, p))
for i in range(N - p):
    for j in range(p):
        Y[i, j] = series[(N-1)-i-1-j]

# Greedy

chosen_idxs = list()
for k in range(nonzero_vals):
    # print(f'Step {k}:')
    min_error = np.inf
    new_chosen_idx = None
    for idx in range(p):
        if idx not in chosen_idxs:
            searched_idxs = chosen_idxs.copy()
            searched_idxs.append(idx)
            # print(f'Searched indexes: {searched_idxs}')
            Y_curr = Y[:, searched_idxs]            

            Gamma = np.dot(Y_curr.T, Y_curr)
            series_rev = series[::-1]
            y = series_rev[:-p]
            gamma = np.dot(Y_curr.T, y)
            x_star_curr = np.dot(np.linalg.inv(Gamma), gamma)
            
            error = np.linalg.norm(y - np.dot(Y_curr, x_star_curr))

            if error < min_error:
                min_error = error
                new_chosen_idx = idx
                params = x_star_curr
    chosen_idxs.append(new_chosen_idx)
    chosen_idxs.sort()

x_star = np.zeros(p)

x_star[chosen_idxs] = params

print(f'Chosen indexes: {chosen_idxs}')
print(f'X_star: {x_star}')

predictions = series[:p].tolist()
while len(predictions) < N:
    predictions.append(np.dot(np.flip(predictions[-p:]), x_star))

plt.plot(x, series, label='Original', color='b')
plt.plot(x, predictions, label='Predicted', color='g', linestyle='dashed')
plt.legend(loc='upper right')
plt.title(f'Predicted series from first {p} samples')
plt.savefig(fname=f'Lab10/ex2_fig1.pdf')
plt.show()

print(f'Is stationary?: {is_stationary(x_star)}')

# L1 norm

series_rev = series[::-1]
y = series_rev[:-p]

A = matrix(Y)
b = matrix(y, (len(y), 1))
# print(A.size, b.size)

x = l1regls(A, b)
x_sparse = np.asarray(x).squeeze()
idx = np.argsort(np.abs(x_sparse))[: -nonzero_vals]
# print(idx, idx.shape)
x_sparse[idx] = 0

print(f'With L1 norm: {x_sparse}')

predictions = series[:p].tolist()
while len(predictions) < N:
    predictions.append(np.dot(np.flip(predictions[-p:]), x_sparse))

x = np.linspace(0, 10, N)

plt.plot(x, series, label='Original', color='b')
plt.plot(x, predictions, label='Predicted', color='g', linestyle='dashed')
plt.legend(loc='upper right')
plt.title(f'Predicted series from first {p} samples')
plt.savefig(fname=f'Lab10/ex2_fig2.pdf')
plt.show()

print(f'Is stationary?: {is_stationary(x_sparse)}')

# Try series with no trend - should be stationary

coeffs = np.polyfit(x, series, deg=2)
trend_est = np.polyval(coeffs, x)
series_detrended = series - trend_est

p = 50
model = AutoReg(series_detrended, lags=p)
fit = model.fit()
x_sparse = fit.params[1:]
mu = fit.params[0]   

predictions_detrended = series_detrended[:p].tolist()

for i in range(p, N):
    pred = mu + np.dot(np.flip(predictions_detrended[-p:]), x_sparse)
    predictions_detrended.append(pred)

predictions_detrended = np.array(predictions_detrended)
predictions_full = predictions_detrended + trend_est

plt.plot(x, series, label='Original', color='b')
plt.plot(x, predictions_full, label='Predicted', color='g', linestyle='dashed')
plt.legend(loc='upper right')
plt.title(f'Predicted series from first {p} samples')
plt.savefig(fname=f'Lab10/ex2_fig2.pdf')
plt.show()

print(f'Is stationary?: {is_stationary(x_sparse)}')