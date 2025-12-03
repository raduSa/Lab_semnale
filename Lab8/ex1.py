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

plt.subplot(4, 1, 1)
plt.title('Trend')
plt.plot(x, trend)

plt.subplot(4, 1, 2)
plt.title('Seasonal')
plt.plot(x, seasonal)

plt.subplot(4, 1, 3)
plt.title('Noise')
plt.plot(x, noise)

plt.subplot(4, 1, 4)
plt.title('Series')
plt.plot(x, series)

plt.tight_layout()
plt.savefig(fname=f'Lab8/ex1_fig1.pdf')
plt.show()

p = 300
Y = np.zeros((N-p, p))
for i in range(N - p):
    for j in range(p):
        Y[i, j] = series[(N-1)-i-1-j]

# print(Y[:10], series[-10:])

Gamma = np.dot(Y.T, Y)
series_rev = series[::-1]
gamma = np.dot(Y.T, series_rev[:-p])
x_star = np.dot(np.linalg.inv(Gamma), gamma)

# print(zip(Y.T, series[p:]))

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[len(x)-p-1:len(x)-1]

print(f'gamma = Y_T * y :{gamma}')
print(f'gamma using np.correlate(signal, signal): {autocorr(series[p:])[::-1]}')

# check reconstructed series is correct
predictions = np.dot(Y, x_star)

plt.plot(x, series, label='Original', color='b')
plt.plot(x[p:], predictions[::-1], label='Predicted', color='g', linestyle='dashed')
plt.legend(loc='upper right')
plt.title('Check reconstructed series')
plt.savefig(fname=f'Lab8/ex1_fig2.pdf')
plt.show()

# predict entire series
predictions = series[:p].tolist()
while len(predictions) < N:
    predictions.append(np.dot(np.flip(predictions[-p:]), x_star))

plt.plot(x, series, label='Original', color='b')
plt.plot(x, predictions, label='Predicted', color='g', linestyle='dashed')
plt.legend(loc='upper right')
plt.title(f'Predicted series from first {p} samples')
plt.savefig(fname=f'Lab8/ex1_fig3.pdf')
plt.show()

# get best next guess model

fold_size = 100
folds = [series[i:i+fold_size+1] for i in range(0, len(series), fold_size)]

best_res = float('inf')
best_params = (None, None)
for p in [2, 10, 20, 30, 40, 50, 100, 200]:
    for m in [200, 300, 400, 500, 600, 800]:   
        MSE = 0
        for last_train_index in range(m, N):
            train_samples = series[last_train_index - m : last_train_index]
            prediction_sample = series[last_train_index]
            Y = np.zeros((m-p, p))
            for i in range(m - p):
                for j in range(p):
                    Y[i, j] = train_samples[(m-1)-i-1-j]
            Gamma = np.dot(Y.T, Y)
            train_rev = train_samples[::-1]
            gamma = np.dot(Y.T, train_rev[:-p])
            x_star = np.dot(np.linalg.inv(Gamma), gamma)

            next_step_prediction = np.dot(Y[0], x_star)
            MSE += np.pow(prediction_sample - next_step_prediction, 2)
        print(f'For p={p}, m={m}, got: {MSE}')
        if MSE < best_res:
            best_res = MSE
            best_params = (p, m)

print(f'Best params: {best_params}')